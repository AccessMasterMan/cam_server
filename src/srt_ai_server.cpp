// srt_ai_server.cpp - SRT AI Face Detection Server
// Fixed Architecture: Main (GIL Release) -> Worker (GIL Acquire + Model Init)

#include <iostream>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <queue>
#include <condition_variable>

#include <pybind11/embed.h>
#include <srt/srt.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>

#include "python_bridge.h"

namespace py = pybind11;
using namespace FaceStreaming;

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running(true);
std::atomic<bool> g_pipelines_ready(false);
std::atomic<bool> g_shutdown_started(false);

// ============================================================================
// Client & Queue Definitions
// ============================================================================

struct ClientInfo {
    SRTSOCKET socket;
    std::string ip;
    int port;
    int client_num;
    std::atomic<bool> active;
    std::atomic<bool> is_sender;
    std::thread thread;
    
    ClientInfo() : socket(SRT_INVALID_SOCK), port(0), client_num(0), 
                   active(false), is_sender(false) {}
};

std::vector<ClientInfo*> g_clients;
std::mutex g_clients_mutex;

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> shutdown_;
    
public:
    explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size), shutdown_(false) {}
    ~ThreadSafeQueue() { shutdown(); }
    void shutdown() { shutdown_ = true; cv_.notify_all(); }
    bool push_drop_oldest(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) return false;
        if (queue_.size() >= max_size_) queue_.pop();
        queue_.push(item);
        cv_.notify_one();
        return true;
    }
    bool pop(T& item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                          [this] { return !queue_.empty() || shutdown_; })) return false;
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
};

ThreadSafeQueue<cv::Mat> g_decoded_frames(5);
ThreadSafeQueue<std::vector<uint8_t>> g_encoded_packets(30);

// ============================================================================
// GStreamer Components
// ============================================================================

struct GstPipelineWrapper {
    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;
    GstElement* appsink = nullptr;
    std::mutex mutex;
    ~GstPipelineWrapper() { cleanup(); }
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex);
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        if (appsrc) { gst_object_unref(appsrc); appsrc = nullptr; }
        if (appsink) { gst_object_unref(appsink); appsink = nullptr; }
    }
};

GstPipelineWrapper g_decoder;
GstPipelineWrapper g_encoder;

GstFlowReturn on_decoded_frame(GstAppSink* appsink, gpointer) {
    if (!g_running || g_shutdown_started) return GST_FLOW_EOS;
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo video_info;
    gst_video_info_from_caps(&video_info, caps);
    int width = GST_VIDEO_INFO_WIDTH(&video_info);
    int height = GST_VIDEO_INFO_HEIGHT(&video_info);
    
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    cv::Mat frame(height, width, CV_8UC3);
    memcpy(frame.data, map.data, map.size);
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    
    g_decoded_frames.push_drop_oldest(frame);
    return GST_FLOW_OK;
}

GstFlowReturn on_encoded_packet(GstAppSink* appsink, gpointer) {
    if (!g_running || g_shutdown_started) return GST_FLOW_EOS;
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    std::vector<uint8_t> packet(map.data, map.data + map.size);
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    g_encoded_packets.push_drop_oldest(packet);
    return GST_FLOW_OK;
}

bool setup_decoder() {
    std::cout << "[Decoder] Setting up H.264 decoder pipeline..." << std::endl;
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-h264,stream-format=byte-stream ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_decoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) { std::cerr << error->message << std::endl; return false; }
    
    g_decoder.appsrc = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "src");
    g_decoder.appsink = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "sink");
    
    g_object_set(G_OBJECT(g_decoder.appsrc), "stream-type", GST_APP_STREAM_TYPE_STREAM, "is-live", TRUE, nullptr);
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_decoded_frame;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_decoder.appsink), &callbacks, nullptr, nullptr);
    gst_app_sink_set_drop(GST_APP_SINK(g_decoder.appsink), TRUE);
    
    gst_element_set_state(g_decoder.pipeline, GST_STATE_PLAYING);
    std::cout << "[Decoder] ✓ Ready" << std::endl;
    return true;
}

bool setup_encoder(int w, int h, int fps) {
    std::cout << "[Encoder] Setting up H.264 encoder pipeline..." << std::endl;
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! "
        "video/x-h264,profile=high ! mpegtsmux ! appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_encoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) { std::cerr << error->message << std::endl; return false; }
    
    g_encoder.appsrc = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "src");
    g_encoder.appsink = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "sink");
    
    g_object_set(G_OBJECT(g_encoder.appsrc), "stream-type", GST_APP_STREAM_TYPE_STREAM, "is-live", TRUE, nullptr);
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_encoded_packet;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_encoder.appsink), &callbacks, nullptr, nullptr);
    
    gst_element_set_state(g_encoder.pipeline, GST_STATE_PLAYING);
    std::cout << "[Encoder] ✓ Ready" << std::endl;
    return true;
}

bool push_frame_to_encoder(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(g_encoder.mutex);
    if (!g_encoder.appsrc || !g_running) return false;
    
    size_t size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame.data, size);
    gst_buffer_unmap(buffer, &map);
    
    gst_app_src_push_buffer(GST_APP_SRC(g_encoder.appsrc), buffer);
    return true;
}

bool push_data_to_decoder(const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(g_decoder.mutex);
    if (!g_decoder.appsrc || !g_running) return false;
    
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, data, size);
    gst_buffer_unmap(buffer, &map);
    
    gst_app_src_push_buffer(GST_APP_SRC(g_decoder.appsrc), buffer);
    return true;
}

// ============================================================================
// AI Processing Thread (FIXED: Init moved here)
// ============================================================================

void ai_processing_thread(PythonBridge& py_bridge) {
    std::cout << "[AI Thread] Started. Initializing AI Engine on this thread..." << std::endl;

    // --- CRITICAL FIX START ---
    // Initialize AI here (inside the worker thread).
    // This creates the CUDA context and TensorRT engine bound to THIS thread.
    // The PythonBridge::initialize method now acquires the GIL internally.
    if (!py_bridge.initialize(640)) {
        std::cerr << "[AI Thread] FATAL: Failed to initialize AI model." << std::endl;
        g_running = false; // Stop the server
        return;
    }
    std::cout << "[AI Thread] Model initialized successfully. Starting loop." << std::endl;
    // --- CRITICAL FIX END ---
    
    while (!g_pipelines_ready && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::vector<DetectionResult> detections;
    int frames_processed = 0;
    
    while (g_running && !g_shutdown_started) {
        cv::Mat frame;
        
        // Wait up to 100ms for a frame
        if (!g_decoded_frames.pop(frame, 100)) {
            continue; 
        }
        
        if (frame.empty()) continue;
        
        detections.clear();
        // Detect faces (uses GIL internally)
        if (!py_bridge.detectFaces(frame, detections)) {
            // Log error but don't crash
            // std::cerr << "[AI] Error: " << py_bridge.getLastError() << std::endl;
        }
        
        drawDetections(frame, detections);
        push_frame_to_encoder(frame);
        
        frames_processed++;
        if (frames_processed % 150 == 0) std::cout << "[AI] Processed " << frames_processed << " frames" << std::endl;
    }
    
    std::cout << "[AI Thread] Stopped" << std::endl;
}

// ============================================================================
// Broadcast & Client Handling (Unchanged Logic)
// ============================================================================

void broadcast_thread() {
    while (!g_pipelines_ready && g_running) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "[Broadcast Thread] Ready" << std::endl;
    
    while (g_running && !g_shutdown_started) {
        std::vector<uint8_t> packet;
        if (!g_encoded_packets.pop(packet, 100)) continue;
        
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (auto* client : g_clients) {
            if (!client->active || client->is_sender) continue;
            if (srt_send(client->socket, reinterpret_cast<const char*>(packet.data()), packet.size()) == SRT_ERROR) {
                client->active = false;
            }
        }
    }
}

void handleClient(ClientInfo* client_info) {
    SRTSOCKET sock = client_info->socket;
    std::vector<uint8_t> buffer(65536);
    int recv_timeout = 100;
    srt_setsockopt(sock, 0, SRTO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout));
    
    bool role_determined = false;
    
    while (g_running && client_info->active) {
        int ret = srt_recv(sock, reinterpret_cast<char*>(buffer.data()), buffer.size());
        if (ret > 0) {
            if (!role_determined) { client_info->is_sender = true; role_determined = true; }
            push_data_to_decoder(buffer.data(), ret);
        } else if (ret == SRT_ERROR) {
            if (srt_getlasterror(nullptr) != SRT_EASYNCRCV && srt_getlasterror(nullptr) != SRT_ETIMEOUT) break;
        }
    }
    srt_close(sock);
    client_info->active = false;
}

void signalHandler(int) {
    g_running = false;
    g_shutdown_started = true;
    g_decoded_frames.shutdown();
    g_encoded_packets.shutdown();
}

int main() {
    signal(SIGINT, signalHandler);
    gst_init(nullptr, nullptr);
    srt_startup();
    
    std::cout << "Starting SRT AI Server..." << std::endl;
    
    // 1. Start Python Interpreter (Holds GIL initially)
    py::scoped_interpreter python_guard{};
    
    // 2. Create Bridge but DO NOT Initialize yet
    PythonBridge py_bridge;
    
    // 3. IMMEDIATELY Release GIL so worker thread can pick it up
    py_bridge.releaseGIL();
    
    // 4. Start GStreamer
    setup_decoder();
    setup_encoder(640, 640, 30);
    g_pipelines_ready = true;
    
    // 5. Start Threads (AI thread will init model)
    std::thread ai_thread(ai_processing_thread, std::ref(py_bridge));
    std::thread broadcast_thr(broadcast_thread);
    
    // 6. Network Setup
    SRTSOCKET listen_socket = srt_create_socket();
    int live = SRTT_LIVE; srt_setsockopt(listen_socket, 0, SRTO_TRANSTYPE, &live, sizeof(live));
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons(9000); sa.sin_addr.s_addr = INADDR_ANY;
    srt_bind(listen_socket, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    srt_listen(listen_socket, 5);
    
    int to = 500; srt_setsockopt(listen_socket, 0, SRTO_RCVTIMEO, &to, sizeof(to));
    
    std::cout << "[Server] Listening on 9000..." << std::endl;
    
    while (g_running) {
        sockaddr_storage client_addr;
        int addr_len = sizeof(client_addr);
        SRTSOCKET client_socket = srt_accept(listen_socket, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
        
        if (client_socket != SRT_INVALID_SOCK) {
            auto* client = new ClientInfo();
            client->socket = client_socket;
            client->active = true;
            {
                std::lock_guard<std::mutex> lock(g_clients_mutex);
                g_clients.push_back(client);
            }
            client->thread = std::thread(handleClient, client);
            client->thread.detach();
        }
    }
    
    // Cleanup
    if (ai_thread.joinable()) ai_thread.join();
    if (broadcast_thr.joinable()) broadcast_thr.join();
    
    // Restore GIL before destruction
    py_bridge.restoreGIL();
    
    srt_close(listen_socket);
    srt_cleanup();
    gst_deinit();
    return 0;
}

