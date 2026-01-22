// python_bridge.cpp - pybind11-based Python Bridge Implementation
// 
// CRITICAL FIX: Uses PyEval_SaveThread/RestoreThread directly instead of
// storing gil_scoped_release in unique_ptr, which causes crashes.
//
// Correct pattern from pybind11 GitHub discussions:
// 1. Initialize interpreter
// 2. Do initialization work  
// 3. PyEval_SaveThread() to release GIL
// 4. Worker threads use gil_scoped_acquire
// 5. PyEval_RestoreThread() before shutdown

#include "python_bridge.h"
#include <iostream>
#include <chrono>

namespace FaceStreaming {

PythonBridge::PythonBridge()
    : initialized_(false)
    , gil_released_(false)
    , det_size_(640)
    , module_(nullptr)
    , detect_func_(nullptr)
    , saved_thread_state_(nullptr)
    , total_calls_(0)
    , total_time_ms_(0.0)
{
}

PythonBridge::~PythonBridge() {
    // Restore GIL if it was released (should be called explicitly before this)
    if (gil_released_ && saved_thread_state_) {
        std::cerr << "[PythonBridge] WARNING: GIL not restored before destruction!" << std::endl;
        PyEval_RestoreThread(saved_thread_state_);
        gil_released_ = false;
    }
    
    // Now safe to destroy Python objects
    detect_func_.reset();
    module_.reset();
}

bool PythonBridge::initialize(int det_size) {
    det_size_ = det_size;
    
    std::cout << "[PythonBridge] Initializing with pybind11..." << std::endl;
    
    try {
        // Add current directory and src to Python path
        std::cout << "[PythonBridge] Adding paths to sys.path..." << std::endl;
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append(".");
        path.append("./src");
        std::cout << "[PythonBridge]   Added: . and ./src" << std::endl;
        
        // Import the ai_worker module
        std::cout << "[PythonBridge] Loading ai_worker module..." << std::endl;
        module_ = std::make_unique<py::module_>(py::module_::import("ai_worker"));
        std::cout << "[PythonBridge] ✓ ai_worker module loaded" << std::endl;
        
        // Get the initialize function and call it
        std::cout << "[PythonBridge] Calling ai_worker.initialize(det_size=" << det_size_ << ")..." << std::endl;
        py::object init_func = module_->attr("initialize");
        init_func(det_size_);
        std::cout << "[PythonBridge] ✓ ai_worker.initialize() completed" << std::endl;
        
        // Get the detect_faces function
        detect_func_ = std::make_unique<py::object>(module_->attr("detect_faces"));
        std::cout << "[PythonBridge] ✓ detect_faces function acquired" << std::endl;
        
        std::cout << "[PythonBridge] ✓ AI engine initialized successfully!" << std::endl;
        
        initialized_ = true;
        return true;
        
    } catch (const py::error_already_set& e) {
        last_error_ = std::string("Python error: ") + e.what();
        std::cerr << "[PythonBridge] " << last_error_ << std::endl;
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        std::cerr << "[PythonBridge] " << last_error_ << std::endl;
        return false;
    }
}

void PythonBridge::releaseGIL() {
    if (!initialized_) {
        std::cerr << "[PythonBridge] Cannot release GIL - not initialized" << std::endl;
        return;
    }
    
    if (gil_released_) {
        std::cout << "[PythonBridge] GIL already released" << std::endl;
        return;
    }
    
    std::cout << "[PythonBridge] Releasing GIL for multi-threaded access..." << std::endl;
    
    // CRITICAL: Use PyEval_SaveThread directly, NOT gil_scoped_release in unique_ptr
    // This is the correct pattern from pybind11 documentation
    saved_thread_state_ = PyEval_SaveThread();
    gil_released_ = true;
    
    std::cout << "[PythonBridge] ✓ GIL released (thread state saved)" << std::endl;
}

void PythonBridge::restoreGIL() {
    if (!gil_released_ || !saved_thread_state_) {
        std::cout << "[PythonBridge] GIL not released or already restored" << std::endl;
        return;
    }
    
    std::cout << "[PythonBridge] Restoring GIL..." << std::endl;
    PyEval_RestoreThread(saved_thread_state_);
    saved_thread_state_ = nullptr;
    gil_released_ = false;
    std::cout << "[PythonBridge] ✓ GIL restored" << std::endl;
}

py::array_t<uint8_t> PythonBridge::cvMatToNumpy(const cv::Mat& frame) {
    // Create a numpy array that shares memory with cv::Mat
    // Shape: (height, width, channels)
    std::vector<ssize_t> shape = {frame.rows, frame.cols, frame.channels()};
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(frame.step[0]),  // bytes per row
        static_cast<ssize_t>(frame.step[1]),  // bytes per pixel
        static_cast<ssize_t>(frame.elemSize1()) // bytes per channel
    };
    
    // Create array without copying data (shares memory with cv::Mat)
    return py::array_t<uint8_t>(
        shape,
        strides,
        frame.data,
        py::none()  // No base object - we manage the lifetime
    );
}

bool PythonBridge::parseResults(const py::list& py_result, std::vector<DetectionResult>& results) {
    results.clear();
    results.reserve(py::len(py_result));
    
    for (size_t i = 0; i < py::len(py_result); ++i) {
        py::dict face_dict = py_result[i].cast<py::dict>();
        
        DetectionResult det;
        
        // Parse bbox
        if (face_dict.contains("bbox")) {
            py::list bbox = face_dict["bbox"].cast<py::list>();
            if (py::len(bbox) == 4) {
                int x1 = bbox[0].cast<int>();
                int y1 = bbox[1].cast<int>();
                int x2 = bbox[2].cast<int>();
                int y2 = bbox[3].cast<int>();
                det.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            }
        }
        
        // Parse landmarks
        if (face_dict.contains("kps")) {
            py::list kps = face_dict["kps"].cast<py::list>();
            det.landmarks.reserve(py::len(kps));
            
            for (size_t j = 0; j < py::len(kps); ++j) {
                py::list point = kps[j].cast<py::list>();
                if (py::len(point) == 2) {
                    float x = point[0].cast<float>();
                    float y = point[1].cast<float>();
                    det.landmarks.emplace_back(x, y);
                }
            }
        }
        
        // Parse confidence
        if (face_dict.contains("conf")) {
            det.confidence = face_dict["conf"].cast<float>();
        } else {
            det.confidence = 1.0f;
        }
        
        results.push_back(det);
    }
    
    return true;
}

bool PythonBridge::detectFaces(const cv::Mat& frame, std::vector<DetectionResult>& results) {
    if (!initialized_) {
        last_error_ = "PythonBridge not initialized";
        return false;
    }
    
    if (frame.empty()) {
        last_error_ = "Input frame is empty";
        return false;
    }
    
    if (!frame.isContinuous()) {
        last_error_ = "Frame must be continuous";
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Acquire GIL for Python operations
        // This works correctly when main thread has called PyEval_SaveThread
        py::gil_scoped_acquire acquire;
        
        // Convert cv::Mat to numpy array (zero-copy)
        py::array_t<uint8_t> np_frame = cvMatToNumpy(frame);
        
        // Call detect_faces
        py::object py_result = (*detect_func_)(np_frame);
        
        // Parse results
        py::list result_list = py_result.cast<py::list>();
        if (!parseResults(result_list, results)) {
            return false;
        }
        
    } catch (const py::error_already_set& e) {
        last_error_ = std::string("Python error: ") + e.what();
        std::cerr << "[PythonBridge] " << last_error_ << std::endl;
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        std::cerr << "[PythonBridge] " << last_error_ << std::endl;
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Update statistics
    total_calls_++;
    total_time_ms_ += elapsed_ms;
    
    // Log stats periodically
    if (total_calls_ % 100 == 0) {
        double avg_ms = total_time_ms_ / total_calls_;
        std::cout << "[PythonBridge] Stats: " << total_calls_ << " calls, avg " 
                  << avg_ms << " ms/frame" << std::endl;
    }
    
    return true;
}

void drawDetections(cv::Mat& frame, 
                   const std::vector<DetectionResult>& results,
                   const cv::Scalar& box_color,
                   const cv::Scalar& landmark_color) {
    for (const auto& det : results) {
        // Draw bounding box
        cv::rectangle(frame, det.bbox, box_color, 2);
        
        // Draw landmarks
        for (const auto& pt : det.landmarks) {
            cv::circle(frame, pt, 3, landmark_color, -1);
        }
        
        // Draw confidence
        if (det.confidence > 0.0f && det.confidence < 1.0f) {
            std::string conf_text = std::to_string(int(det.confidence * 100)) + "%";
            cv::putText(frame, conf_text, 
                       cv::Point(det.bbox.x, det.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
        }
    }
}

} // namespace FaceStreaming