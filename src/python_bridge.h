// python_bridge.h - pybind11-based Python Bridge for Face Detection
// Uses pybind11 for automatic memory management and GIL handling

#ifndef PYTHON_BRIDGE_H
#define PYTHON_BRIDGE_H

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace py = pybind11;

namespace FaceStreaming {

/**
 * Detection result structure
 */
struct DetectionResult {
    cv::Rect bbox;
    std::vector<cv::Point2f> landmarks;
    float confidence;
};

/**
 * PythonBridge - Thread-safe Python/C++ interface for face detection
 * 
 * Uses pybind11 for proper GIL management and memory safety.
 * 
 * Usage pattern:
 * 1. Create PythonBridge in main thread
 * 2. Call initialize() - this starts the interpreter and loads the model
 * 3. Call releaseGIL() to allow other threads to use Python
 * 4. Worker threads can now call detectFaces() safely
 */
class PythonBridge {
public:
    PythonBridge();
    ~PythonBridge();
    
    // Non-copyable
    PythonBridge(const PythonBridge&) = delete;
    PythonBridge& operator=(const PythonBridge&) = delete;
    
    /**
     * Initialize Python interpreter and load AI model
     * Must be called from main thread before any other operations
     * @param det_size Detection input size (640 recommended)
     * @return true on success
     */
    bool initialize(int det_size = 640);
    
    /**
     * Release GIL so other threads can use Python
     * Must be called after initialize() and before worker threads start
     */
    void releaseGIL();
    
    /**
     * Detect faces in a frame (thread-safe)
     * Automatically acquires/releases GIL
     * @param frame Input BGR frame
     * @param results Output detection results
     * @return true on success
     */
    bool detectFaces(const cv::Mat& frame, std::vector<DetectionResult>& results);
    
    bool isInitialized() const { return initialized_; }
    std::string getLastError() const { return last_error_; }

private:
    // Convert cv::Mat to pybind11 numpy array
    py::array_t<uint8_t> cvMatToNumpy(const cv::Mat& frame);
    
    // Parse Python detection results
    bool parseResults(const py::list& py_result, std::vector<DetectionResult>& results);
    
    bool initialized_;
    int det_size_;
    std::string last_error_;
    
    // Python objects (must be destroyed before interpreter finalization)
    std::unique_ptr<py::module_> module_;
    std::unique_ptr<py::object> detect_func_;
    
    // GIL release guard (kept alive to keep GIL released)
    std::unique_ptr<py::gil_scoped_release> gil_release_;
    
    // Statistics
    uint64_t total_calls_;
    double total_time_ms_;
};

/**
 * Draw detection results on frame
 */
void drawDetections(cv::Mat& frame, 
                   const std::vector<DetectionResult>& results,
                   const cv::Scalar& box_color = cv::Scalar(0, 255, 0),
                   const cv::Scalar& landmark_color = cv::Scalar(0, 0, 255));

} // namespace FaceStreaming

#endif // PYTHON_BRIDGE_H