// python_bridge.cpp - Zero-Copy Python Bridge Implementation (PRODUCTION FIXED)

#include "python_bridge.h"
#include <iostream>
#include <sstream>
#include <chrono>

namespace FaceStreaming {

PythonBridge::PythonBridge()
    : module_(nullptr)
    , detect_func_(nullptr)
    , initialized_(false)
    , det_size_(640)
    , total_calls_(0)
    , total_time_ms_(0.0)
{
}

PythonBridge::~PythonBridge() {
    if (initialized_) {
        GILGuard gil;
        
        Py_XDECREF(detect_func_);
        Py_XDECREF(module_);
        
        if (Py_IsInitialized()) {
            Py_Finalize();
        }
    }
}

bool PythonBridge::initialize(int det_size) {
    det_size_ = det_size;
    
    std::cout << "[PythonBridge] Initializing Python environment..." << std::endl;
    
    Py_Initialize();
    
    if (!Py_IsInitialized()) {
        last_error_ = "Failed to initialize Python interpreter";
        return false;
    }
    
    // CRITICAL FIX: Initialize threading support and release GIL
    // This allows other threads to call Python code safely
    if (!PyEval_ThreadsInitialized()) {
        PyEval_InitThreads();
    }
    
    import_array1(false);
    
    std::cout << "[PythonBridge] Python " << Py_GetVersion() << std::endl;
    std::cout << "[PythonBridge] NumPy C API imported" << std::endl;
    
    {
        GILGuard gil;
        
        PyObject* sys_path = PySys_GetObject((char*)"path");
        PyObject* cwd = PyUnicode_FromString("./src");
        PyList_Append(sys_path, cwd);
        Py_DECREF(cwd);
        
        PyObject* parent = PyUnicode_FromString(".");
        PyList_Append(sys_path, parent);
        Py_DECREF(parent);
    }
    
    {
        GILGuard gil;
        
        std::cout << "[PythonBridge] Loading ai_worker module..." << std::endl;
        
        module_ = PyImport_ImportModule("ai_worker");
        if (!module_) {
            setPythonError();
            PyErr_Print();
            return false;
        }
        
        PyObject* init_func = PyObject_GetAttrString(module_, "initialize");
        if (!init_func || !PyCallable_Check(init_func)) {
            last_error_ = "ai_worker.initialize not found or not callable";
            Py_XDECREF(init_func);
            return false;
        }
        
        std::cout << "[PythonBridge] Initializing InspireFace (det_size=" << det_size_ << ")..." << std::endl;
        
        PyObject* args = Py_BuildValue("(i)", det_size_);
        PyObject* result = PyObject_CallObject(init_func, args);
        Py_DECREF(args);
        Py_DECREF(init_func);
        
        if (!result) {
            setPythonError();
            PyErr_Print();
            return false;
        }
        
        Py_DECREF(result);
        
        detect_func_ = PyObject_GetAttrString(module_, "detect_faces");
        if (!detect_func_ || !PyCallable_Check(detect_func_)) {
            last_error_ = "ai_worker.detect_faces not found or not callable";
            Py_XDECREF(detect_func_);
            detect_func_ = nullptr;
            return false;
        }
        
        std::cout << "[PythonBridge] ✓ AI engine initialized successfully!" << std::endl;
    }
    
    // CRITICAL FIX: Release GIL so other threads can use Python
    PyEval_SaveThread();
    
    initialized_ = true;
    return true;
}

PyObject* PythonBridge::cvMatToNumPy(const cv::Mat& frame) {
    if (!frame.isContinuous()) {
        last_error_ = "Frame must be continuous for zero-copy conversion";
        return nullptr;
    }
    
    if (frame.empty()) {
        last_error_ = "Frame is empty";
        return nullptr;
    }
    
    npy_intp dims[3] = {frame.rows, frame.cols, frame.channels()};
    
    // CRITICAL FIX: Make a COPY of the data to avoid use-after-free
    // Zero-copy is unsafe when frames are being processed asynchronously
    PyObject* array = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!array) {
        last_error_ = "Failed to create NumPy array";
        return nullptr;
    }
    
    void* array_data = PyArray_DATA((PyArrayObject*)array);
    memcpy(array_data, frame.data, frame.total() * frame.elemSize());
    
    return array;
}

bool PythonBridge::parseResults(PyObject* py_result, std::vector<DetectionResult>& results) {
    results.clear();
    
    if (!PyList_Check(py_result)) {
        last_error_ = "Python result is not a list";
        return false;
    }
    
    Py_ssize_t num_faces = PyList_Size(py_result);
    results.reserve(num_faces);
    
    for (Py_ssize_t i = 0; i < num_faces; ++i) {
        PyObject* face_dict = PyList_GetItem(py_result, i);
        
        if (!PyDict_Check(face_dict)) {
            continue;
        }
        
        DetectionResult det;
        
        PyObject* bbox_list = PyDict_GetItemString(face_dict, "bbox");
        if (bbox_list && PyList_Check(bbox_list) && PyList_Size(bbox_list) == 4) {
            int x1 = PyLong_AsLong(PyList_GetItem(bbox_list, 0));
            int y1 = PyLong_AsLong(PyList_GetItem(bbox_list, 1));
            int x2 = PyLong_AsLong(PyList_GetItem(bbox_list, 2));
            int y2 = PyLong_AsLong(PyList_GetItem(bbox_list, 3));
            det.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        }
        
        PyObject* kps_list = PyDict_GetItemString(face_dict, "kps");
        if (kps_list && PyList_Check(kps_list)) {
            Py_ssize_t num_kps = PyList_Size(kps_list);
            det.landmarks.reserve(num_kps);
            
            for (Py_ssize_t j = 0; j < num_kps; ++j) {
                PyObject* point_list = PyList_GetItem(kps_list, j);
                if (PyList_Check(point_list) && PyList_Size(point_list) == 2) {
                    float x = PyFloat_AsDouble(PyList_GetItem(point_list, 0));
                    float y = PyFloat_AsDouble(PyList_GetItem(point_list, 1));
                    det.landmarks.emplace_back(x, y);
                }
            }
        }
        
        PyObject* conf_obj = PyDict_GetItemString(face_dict, "conf");
        if (conf_obj) {
            det.confidence = PyFloat_AsDouble(conf_obj);
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
    
    auto start = std::chrono::high_resolution_clock::now();
    
    GILGuard gil;
    
    PyObject* np_frame = cvMatToNumPy(frame);
    if (!np_frame) {
        return false;
    }
    
    PyObject* args = PyTuple_Pack(1, np_frame);
    if (!args) {
        Py_DECREF(np_frame);
        last_error_ = "Failed to create argument tuple";
        return false;
    }
    
    PyObject* py_result = PyObject_CallObject(detect_func_, args);
    
    Py_DECREF(args);
    Py_DECREF(np_frame);
    
    if (!py_result) {
        setPythonError();
        PyErr_Print();
        return false;
    }
    
    bool success = parseResults(py_result, results);
    Py_DECREF(py_result);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    total_calls_++;
    total_time_ms_ += elapsed_ms;
    
    if (total_calls_ % 100 == 0) {
        double avg_ms = total_time_ms_ / total_calls_;
        std::cout << "[PythonBridge] Stats: " << total_calls_ << " calls, avg " 
                  << avg_ms << " ms/frame" << std::endl;
    }
    
    return success;
}

void PythonBridge::setPythonError() {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    
    if (value) {
        PyObject* str = PyObject_Str(value);
        if (str) {
            const char* err_msg = PyUnicode_AsUTF8(str);
            if (err_msg) {
                last_error_ = err_msg;
            }
            Py_DECREF(str);
        }
    }
    
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
}

bool PythonBridge::restart(int det_size) {
    std::cout << "[PythonBridge] Restarting Python interpreter..." << std::endl;
    
    if (initialized_) {
        GILGuard gil;
        Py_XDECREF(detect_func_);
        Py_XDECREF(module_);
        detect_func_ = nullptr;
        module_ = nullptr;
    }
    
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
    
    initialized_ = false;
    total_calls_ = 0;
    total_time_ms_ = 0.0;
    
    return initialize(det_size);
}

void drawDetections(cv::Mat& frame, 
                   const std::vector<DetectionResult>& results,
                   const cv::Scalar& box_color,
                   const cv::Scalar& landmark_color) {
    for (const auto& det : results) {
        cv::rectangle(frame, det.bbox, box_color, 2);
        
        for (const auto& pt : det.landmarks) {
            cv::circle(frame, pt, 3, landmark_color, -1);
        }
        
        if (det.confidence > 0.0f && det.confidence < 1.0f) {
            std::string conf_text = std::to_string(int(det.confidence * 100)) + "%";
            cv::putText(frame, conf_text, 
                       cv::Point(det.bbox.x, det.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
        }
    }
}

} // namespace FaceStreaming