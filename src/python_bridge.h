// python_bridge.h - Zero-Copy Python Bridge for Face Detection

#ifndef PYTHON_BRIDGE_H
#define PYTHON_BRIDGE_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace FaceStreaming {

struct DetectionResult {
    cv::Rect bbox;
    std::vector<cv::Point2f> landmarks;
    float confidence;
};

class GILGuard {
public:
    GILGuard() : gstate_(PyGILState_Ensure()) {}
    ~GILGuard() { PyGILState_Release(gstate_); }
    
    GILGuard(const GILGuard&) = delete;
    GILGuard& operator=(const GILGuard&) = delete;
    
private:
    PyGILState_STATE gstate_;
};

class PythonBridge {
public:
    PythonBridge();
    ~PythonBridge();
    
    PythonBridge(const PythonBridge&) = delete;
    PythonBridge& operator=(const PythonBridge&) = delete;
    
    bool initialize(int det_size = 640);
    bool detectFaces(const cv::Mat& frame, std::vector<DetectionResult>& results);
    bool isInitialized() const { return initialized_; }
    std::string getLastError() const { return last_error_; }
    bool restart(int det_size = 640);

private:
    PyObject* cvMatToNumPy(const cv::Mat& frame);
    bool parseResults(PyObject* py_result, std::vector<DetectionResult>& results);
    void setPythonError();
    
    PyObject* module_;
    PyObject* detect_func_;
    
    bool initialized_;
    int det_size_;
    std::string last_error_;
    
    uint64_t total_calls_;
    double total_time_ms_;
};

void drawDetections(cv::Mat& frame, 
                   const std::vector<DetectionResult>& results,
                   const cv::Scalar& box_color = cv::Scalar(0, 255, 0),
                   const cv::Scalar& landmark_color = cv::Scalar(0, 0, 255));

} // namespace FaceStreaming

#endif // PYTHON_BRIDGE_H