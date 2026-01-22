// srt_ai_server.cpp - PRODUCTION FIXED VERSION
// SRT AI Face Detection Server - Broadcast Architecture
// ONE PORT: Receives H.264 from sender → AI process → Broadcasts processed H.264 to receivers
//
// FIXES APPLIED:
// 1. Python GIL properly released after initialization (in python_bridge.cpp)
// 2. GStreamer element references properly managed
// 3. Thread synchronization improved with condition variables
// 4. Proper cleanup sequence

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
#include <srt/srt.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include "python_bridge.h"

using namespace FaceStreaming;

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running(true);
std::atomic<bool> g_pipelines_ready(false);
std::atomic<bool> g_shutdown_started(false);

// ============================================================================
// Client Management
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

// ============================================================================
// Thread-Safe Queue with Proper Synchronization
// ============================================================================

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    std::atomic<bool> shutdown_;
    
public:
    explicit ThreadSafeQueue(size_t max_size = 10) 
        : max_size_(max_size), shutdown_(false) {}
    
    ~ThreadSafeQueue() {
        shutdown();
    }
    
    void shutdown() {
        shutdown_ = true;
        cv_.notify_all();
        cv_not_full_.notify_all();
    }
    
    bool push(const T& item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait for space if queue is full
        if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                                    [this] { return queue_.size() < max_size_ || shutdown_; })) {
            return false;
        }
        
        if (shutdown_) return false;
        
        queue_.push(item);
        lock.unlock();
        cv_.notify_one();
        return true;
    }
    
    bool push_drop_oldest(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (shutdown_) return false;
        
        // Drop oldest if full (for low-latency scenarios)
        if (queue_.size() >= max_size_) {
            queue_.pop();
        }
        
        queue_.push(item);
        cv_.notify_one();
        return true;
    }
    
    bool pop(T& item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                          [this] { return !queue_.empty() || shutdown_; })) {
            return false;
        }
        
        if (shutdown_ && queue_.empty()) return false;
        if (queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cv_not_full_.notify_one();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }
};

// Global queues
ThreadSafeQueue<cv::Mat> g_decoded_frames(5);
ThreadSafeQueue<std::vector<uint8_t>> g_encoded_packets(30);

// ============================================================================
// GStreamer Pipeline Wrapper with Proper Reference Management
// ============================================================================

struct GstPipelineWrapper {
    GstElement* pipeline;
    GstElement* appsrc;    // Owned reference from gst_bin_get_by_name
    GstElement* appsink;   // Owned reference from gst_bin_get_by_name
    std::mutex mutex;      // Protects access to pipeline elements
    
    GstPipelineWrapper() : pipeline(nullptr), appsrc(nullptr), appsink(nullptr) {}
    
    ~GstPipelineWrapper() {
        cleanup();
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            
            // Wait for state change to complete
            GstState state;
            gst_element_get_state(pipeline, &state, nullptr, GST_SECOND);
        }
        
        // CRITICAL FIX: Unref the elements we got from gst_bin_get_by_name
        // These are OWNED references that we must release
        if (appsrc) {
            gst_object_unref(appsrc);
            appsrc = nullptr;
        }
        
        if (appsink) {
            gst_object_unref(appsink);
            appsink = nullptr;
        }
        
        if (pipeline) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
    }
    
    bool isValid() const {
        return pipeline != nullptr && appsrc != nullptr && appsink != nullptr;
    }
};

GstPipelineWrapper g_decoder;
GstPipelineWrapper g_encoder;

// ============================================================================
// GStreamer Decoder Callback (H.264 → Raw Frames)
// ============================================================================

GstFlowReturn on_decoded_frame(GstAppSink* appsink, gpointer /*user_data*/) {
    if (!g_running || g_shutdown_started) {
        return GST_FLOW_EOS;
    }
    
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    
    if (!buffer || !caps) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    GstVideoInfo video_info;
    if (!gst_video_info_from_caps(&video_info, caps)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    int width = GST_VIDEO_INFO_WIDTH(&video_info);
    int height = GST_VIDEO_INFO_HEIGHT(&video_info);
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    size_t expected_size = width * height * 3;
    
    if (map.size < expected_size) {
        std::cerr << "[Decoder] Buffer size mismatch: " << map.size 
                  << " < " << expected_size << std::endl;
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    // Create OpenCV Mat and copy data
    cv::Mat frame(height, width, CV_8UC3);
    memcpy(frame.data, map.data, expected_size);
    
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    
    // Push to queue, dropping oldest if full (for low latency)
    g_decoded_frames.push_drop_oldest(frame);
    
    return GST_FLOW_OK;
}

// ============================================================================
// GStreamer Encoder Callback (Raw Frames → H.264)
// ============================================================================

GstFlowReturn on_encoded_packet(GstAppSink* appsink, gpointer /*user_data*/) {
    if (!g_running || g_shutdown_started) {
        return GST_FLOW_EOS;
    }
    
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    std::vector<uint8_t> packet(map.data, map.data + map.size);
    
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    
    // Push to queue
    g_encoded_packets.push_drop_oldest(packet);
    
    return GST_FLOW_OK;
}

// ============================================================================
// Setup Decoder Pipeline
// ============================================================================

bool setup_decoder() {
    std::cout << "\n[Decoder] ════════════════════════════════════════" << std::endl;
    std::cout << "[Decoder] Setting up H.264 decoder (LIVE SOURCE)" << std::endl;
    
    std::string pipeline_str = 
        "appsrc name=src format=time do-timestamp=true "
        "caps=video/x-h264,stream-format=byte-stream ! "
        "h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false";
    
    std::cout << "[Decoder] Pipeline: " << pipeline_str << std::endl;
    
    GError* error = nullptr;
    g_decoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Decoder] Failed to create pipeline: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    std::cout << "[Decoder] ✓ Pipeline created" << std::endl;
    
    // Get elements - these return NEW references that we own
    g_decoder.appsrc = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "src");
    g_decoder.appsink = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "sink");
    
    if (!g_decoder.appsrc || !g_decoder.appsink) {
        std::cerr << "[Decoder] Failed to get pipeline elements" << std::endl;
        return false;
    }
    std::cout << "[Decoder] ✓ Elements found" << std::endl;
    
    // Configure appsrc for live streaming
    std::cout << "[Decoder] Configuring appsrc for LIVE streaming..." << std::endl;
    g_object_set(G_OBJECT(g_decoder.appsrc),
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);
    std::cout << "[Decoder]   - stream-type: STREAM" << std::endl;
    std::cout << "[Decoder]   - is-live: TRUE" << std::endl;
    
    // Configure appsink callbacks
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_decoded_frame;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_decoder.appsink), &callbacks, nullptr, nullptr);
    gst_app_sink_set_max_buffers(GST_APP_SINK(g_decoder.appsink), 1);
    gst_app_sink_set_drop(GST_APP_SINK(g_decoder.appsink), TRUE);
    std::cout << "[Decoder] ✓ Appsink configured" << std::endl;
    
    // Set to PLAYING
    std::cout << "[Decoder] Setting to PLAYING..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(g_decoder.pipeline, GST_STATE_PLAYING);
    std::cout << "[Decoder] State change return: " 
              << (ret == GST_STATE_CHANGE_ASYNC ? "ASYNC" : 
                  ret == GST_STATE_CHANGE_SUCCESS ? "SUCCESS" : "FAILURE") << std::endl;
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Decoder] Failed to set pipeline to PLAYING" << std::endl;
        return false;
    }
    
    // Wait for pipeline to reach PLAYING or PAUSED state
    GstState state;
    ret = gst_element_get_state(g_decoder.pipeline, &state, nullptr, 2 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Decoder] Pipeline failed to reach ready state" << std::endl;
        return false;
    }
    
    std::cout << "[Decoder] ✓ Ready (state: " << state << ")" << std::endl;
    std::cout << "[Decoder] ════════════════════════════════════════\n" << std::endl;
    return true;
}

// ============================================================================
// Setup Encoder Pipeline
// ============================================================================

bool setup_encoder(int width, int height, int fps) {
    std::cout << "\n[Encoder] ════════════════════════════════════════" << std::endl;
    std::cout << "[Encoder] Setting up H.264 encoder (LIVE SOURCE)" << std::endl;
    
    std::string pipeline_str = 
        "appsrc name=src format=time do-timestamp=true "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(width) + 
        ",height=" + std::to_string(height) + ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! "
        "video/x-h264,profile=high ! mpegtsmux ! "
        "appsink name=sink emit-signals=true sync=false";
    
    std::cout << "[Encoder] Pipeline: " << pipeline_str << std::endl;
    
    GError* error = nullptr;
    g_encoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Encoder] Failed to create pipeline: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    std::cout << "[Encoder] ✓ Pipeline created" << std::endl;
    
    // Get elements - these return NEW references that we own
    g_encoder.appsrc = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "src");
    g_encoder.appsink = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "sink");
    
    if (!g_encoder.appsrc || !g_encoder.appsink) {
        std::cerr << "[Encoder] Failed to get pipeline elements" << std::endl;
        return false;
    }
    std::cout << "[Encoder] ✓ Elements found" << std::endl;
    
    // Configure appsrc for live streaming
    g_object_set(G_OBJECT(g_encoder.appsrc),
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);
    std::cout << "[Encoder] ✓ Appsrc configured as live source" << std::endl;
    
    // Configure appsink callbacks
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_encoded_packet;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_encoder.appsink), &callbacks, nullptr, nullptr);
    
    // Set to PLAYING
    std::cout << "[Encoder] Setting to PLAYING..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(g_encoder.pipeline, GST_STATE_PLAYING);
    std::cout << "[Encoder] State change: " 
              << (ret == GST_STATE_CHANGE_ASYNC ? "ASYNC" : 
                  ret == GST_STATE_CHANGE_SUCCESS ? "SUCCESS" : "FAILURE") << std::endl;
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Encoder] Failed to set pipeline to PLAYING" << std::endl;
        return false;
    }
    
    // Wait for pipeline to reach ready state
    GstState state;
    ret = gst_element_get_state(g_encoder.pipeline, &state, nullptr, 2 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Encoder] Pipeline failed to reach ready state" << std::endl;
        return false;
    }
    
    std::cout << "[Encoder] ✓ Ready" << std::endl;
    std::cout << "[Encoder] ════════════════════════════════════════\n" << std::endl;
    return true;
}

// ============================================================================
// Push Frame to Encoder (Thread-Safe)
// ============================================================================

bool push_frame_to_encoder(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(g_encoder.mutex);
    
    if (!g_encoder.appsrc || !g_running) {
        return false;
    }
    
    size_t frame_size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
    
    if (!buffer) {
        return false;
    }
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return false;
    }
    
    memcpy(map.data, frame.data, frame_size);
    gst_buffer_unmap(buffer, &map);
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(g_encoder.appsrc), buffer);
    // Note: gst_app_src_push_buffer takes ownership of buffer, don't unref
    
    return (ret == GST_FLOW_OK);
}

// ============================================================================
// Push Data to Decoder (Thread-Safe)
// ============================================================================

bool push_data_to_decoder(const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(g_decoder.mutex);
    
    if (!g_decoder.appsrc || !g_running) {
        return false;
    }
    
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    
    if (!buffer) {
        return false;
    }
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return false;
    }
    
    memcpy(map.data, data, size);
    gst_buffer_unmap(buffer, &map);
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(g_decoder.appsrc), buffer);
    // Note: gst_app_src_push_buffer takes ownership of buffer, don't unref
    
    return (ret == GST_FLOW_OK);
}

// ============================================================================
// AI Processing Thread
// ============================================================================

void ai_processing_thread(PythonBridge& py_bridge) {
    std::cout << "[AI Thread] Started" << std::endl;
    
    // Wait for pipelines to be ready
    while (!g_pipelines_ready && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!g_running) {
        std::cout << "[AI Thread] Stopped before processing" << std::endl;
        return;
    }
    
    std::cout << "[AI Thread] Processing frames..." << std::endl;
    
    std::vector<DetectionResult> detections;
    int frames_processed = 0;
    auto last_stat = std::chrono::steady_clock::now();
    
    while (g_running && !g_shutdown_started) {
        cv::Mat frame;
        
        // Get decoded frame with timeout
        if (!g_decoded_frames.pop(frame, 100)) {
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }
        
        if (frames_processed == 0) {
            std::cout << "[AI Thread] ✓ First frame received: " 
                      << frame.cols << "x" << frame.rows << std::endl;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // AI detection
        detections.clear();
        if (!py_bridge.detectFaces(frame, detections)) {
            std::cerr << "[AI] Detection failed: " << py_bridge.getLastError() << std::endl;
            continue;
        }
        
        // Draw results
        drawDetections(frame, detections);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        frames_processed++;
        
        // Log first few frames
        if (frames_processed <= 3) {
            std::cout << "[AI] Frame #" << frames_processed << " | " 
                      << detections.size() << " face(s) | " << latency_ms << " ms" << std::endl;
        }
        
        // Stats every 5s
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stat).count() >= 5) {
            std::cout << "[AI] Processed " << frames_processed << " frames in last 5s" << std::endl;
            last_stat = now;
            frames_processed = 0;
        }
        
        // Push processed frame to encoder
        if (!push_frame_to_encoder(frame)) {
            if (frames_processed <= 3) {
                std::cerr << "[AI] Failed to push frame to encoder" << std::endl;
            }
        }
    }
    
    std::cout << "[AI Thread] Stopped" << std::endl;
}

// ============================================================================
// Broadcast Thread
// ============================================================================

void broadcast_thread() {
    std::cout << "[Broadcast Thread] Started" << std::endl;
    
    // Wait for pipelines to be ready
    while (!g_pipelines_ready && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!g_running) {
        std::cout << "[Broadcast Thread] Stopped before processing" << std::endl;
        return;
    }
    
    std::cout << "[Broadcast Thread] Ready to broadcast" << std::endl;
    
    int packets_sent = 0;
    
    while (g_running && !g_shutdown_started) {
        std::vector<uint8_t> packet;
        
        // Get encoded packet with timeout
        if (!g_encoded_packets.pop(packet, 100)) {
            continue;
        }
        
        if (packet.empty()) {
            continue;
        }
        
        // Broadcast to all receiver clients
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            for (auto* client : g_clients) {
                if (!client->active || client->is_sender) continue;
                
                int sent = srt_send(client->socket, reinterpret_cast<const char*>(packet.data()), 
                                    static_cast<int>(packet.size()));
                if (sent == SRT_ERROR) {
                    // Don't spam errors, just mark client as inactive
                    client->active = false;
                }
            }
        }
        
        packets_sent++;
        if (packets_sent == 1) {
            std::cout << "[Broadcast] First packet sent (" << packet.size() << " bytes)" << std::endl;
        } else if (packets_sent % 100 == 0) {
            std::cout << "[Broadcast] Sent " << packets_sent << " packets" << std::endl;
        }
    }
    
    std::cout << "[Broadcast Thread] Stopped" << std::endl;
}

// ============================================================================
// Client Handler
// ============================================================================

void handleClient(ClientInfo* client_info) {
    SRTSOCKET client_socket = client_info->socket;
    int client_num = client_info->client_num;
    
    std::cout << "\n[Client #" << client_num << "] Connected: " 
              << client_info->ip << ":" << client_info->port << std::endl;
    
    const int BUFFER_SIZE = 65536;
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    
    // Set receive timeout
    int recv_timeout_ms = 100;
    srt_setsockopt(client_socket, 0, SRTO_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
    
    bool role_determined = false;
    int chunks_received = 0;
    
    auto last_data = std::chrono::steady_clock::now();
    
    while (g_running && client_info->active && !g_shutdown_started) {
        int received = srt_recv(client_socket, reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE);
        
        if (received == SRT_ERROR) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) {
                // Timeout - check if we should mark as receiver
                auto now = std::chrono::steady_clock::now();
                auto idle = std::chrono::duration_cast<std::chrono::seconds>(now - last_data);
                if (idle.count() >= 10 && chunks_received == 0 && !role_determined) {
                    std::cout << "[Client #" << client_num << "] ROLE: RECEIVER (no data sent)" << std::endl;
                    role_determined = true;
                    client_info->is_sender = false;
                }
                continue;
            }
            // Real error
            std::cerr << "[Client #" << client_num << "] Error: " << srt_getlasterror_str() << std::endl;
            break;
        }
        
        if (received > 0) {
            chunks_received++;
            last_data = std::chrono::steady_clock::now();
            
            if (!role_determined) {
                client_info->is_sender = true;
                role_determined = true;
                std::cout << "[Client #" << client_num << "] ROLE: SENDER" << std::endl;
            }
            
            // Push H.264 data to decoder
            if (!push_data_to_decoder(buffer.data(), received)) {
                if (chunks_received <= 3) {
                    std::cerr << "[Client #" << client_num << "] Failed to push to decoder" << std::endl;
                }
            }
            
            if (chunks_received <= 3) {
                std::cout << "[Client #" << client_num << "] Chunk #" << chunks_received 
                          << " received (" << received << " bytes)" << std::endl;
            }
        }
    }
    
    std::cout << "[Client #" << client_num << "] Disconnected (received " 
              << chunks_received << " chunks)" << std::endl;
    
    srt_close(client_socket);
    client_info->active = false;
}

// ============================================================================
// Signal Handler
// ============================================================================

void signalHandler(int /*signum*/) {
    std::cout << "\n[Server] Shutdown signal received" << std::endl;
    g_running = false;
    g_shutdown_started = true;
    
    // Shutdown queues to unblock threads
    g_decoded_frames.shutdown();
    g_encoded_packets.shutdown();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SRT AI BROADCAST SERVER - FIXED VERSION              ║" << std::endl;
    std::cout << "║  Architecture: Sender → AI Processing → Receivers     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Initialize GStreamer
    gst_init(nullptr, nullptr);
    std::cout << "[GStreamer] Initialized" << std::endl;
    
    // Initialize SRT
    if (srt_startup() < 0) {
        std::cerr << "[SRT] Initialization failed" << std::endl;
        return 1;
    }
    std::cout << "[SRT] Initialized" << std::endl;
    
    // Initialize Python AI Bridge
    // CRITICAL: This now properly releases the GIL after initialization
    PythonBridge py_bridge;
    if (!py_bridge.initialize(640)) {
        std::cerr << "[AI] Initialization failed: " << py_bridge.getLastError() << std::endl;
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    std::cout << "[AI] ✓ Ready\n" << std::endl;
    
    // Setup GStreamer pipelines
    if (!setup_decoder()) {
        std::cerr << "[Server] Failed to setup decoder" << std::endl;
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    if (!setup_encoder(640, 640, 30)) {
        std::cerr << "[Server] Failed to setup encoder" << std::endl;
        g_decoder.cleanup();
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    // Small delay to ensure GStreamer pipelines are fully ready
    std::cout << "[Server] Waiting for pipelines to stabilize..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Mark pipelines as ready AFTER setup is complete
    g_pipelines_ready = true;
    std::cout << "[Server] ✓ Pipelines ready" << std::endl;
    
    // Start processing threads
    std::cout << "[Server] Starting processing threads..." << std::endl;
    std::thread ai_thread(ai_processing_thread, std::ref(py_bridge));
    std::thread broadcast_thr(broadcast_thread);
    
    // Give threads time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "[Server] ✓ All threads started\n" << std::endl;
    
    // Create SRT listening socket
    SRTSOCKET listen_socket = srt_create_socket();
    if (listen_socket == SRT_INVALID_SOCK) {
        std::cerr << "[SRT] Failed to create socket" << std::endl;
        g_running = false;
        ai_thread.join();
        broadcast_thr.join();
        return 1;
    }
    
    // Configure SRT socket
    int live_mode = SRTT_LIVE;
    srt_setsockopt(listen_socket, 0, SRTO_TRANSTYPE, &live_mode, sizeof(live_mode));
    
    int latency_ms = 10;
    srt_setsockopt(listen_socket, 0, SRTO_LATENCY, &latency_ms, sizeof(latency_ms));
    
    int tsbpd_mode = 1;
    srt_setsockopt(listen_socket, 0, SRTO_TSBPDMODE, &tsbpd_mode, sizeof(tsbpd_mode));
    
    int rcvbuf = 4000000;
    int sndbuf = 4000000;
    srt_setsockopt(listen_socket, 0, SRTO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    srt_setsockopt(listen_socket, 0, SRTO_SNDBUF, &sndbuf, sizeof(sndbuf));
    
    const int PORT = 9000;
    sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(PORT);
    sa.sin_addr.s_addr = INADDR_ANY;
    
    if (srt_bind(listen_socket, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == SRT_ERROR) {
        std::cerr << "[SRT] Bind failed: " << srt_getlasterror_str() << std::endl;
        srt_close(listen_socket);
        g_running = false;
        g_decoded_frames.shutdown();
        g_encoded_packets.shutdown();
        ai_thread.join();
        broadcast_thr.join();
        return 1;
    }
    
    if (srt_listen(listen_socket, 5) == SRT_ERROR) {
        std::cerr << "[SRT] Listen failed: " << srt_getlasterror_str() << std::endl;
        srt_close(listen_socket);
        g_running = false;
        g_decoded_frames.shutdown();
        g_encoded_packets.shutdown();
        ai_thread.join();
        broadcast_thr.join();
        return 1;
    }
    
    std::cout << "════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  ✓ Listening on port " << PORT << std::endl;
    std::cout << "  ✓ LIVE mode, latency " << latency_ms << " ms" << std::endl;
    std::cout << "  ✓ Waiting for clients..." << std::endl;
    std::cout << "════════════════════════════════════════════════════════\n" << std::endl;
    
    int client_counter = 0;
    
    // Set accept timeout
    int accept_timeout = 500;
    srt_setsockopt(listen_socket, 0, SRTO_RCVTIMEO, &accept_timeout, sizeof(accept_timeout));
    
    // Accept clients
    while (g_running) {
        sockaddr_storage client_addr;
        int addr_len = sizeof(client_addr);
        
        SRTSOCKET client_socket = srt_accept(listen_socket, 
                                              reinterpret_cast<sockaddr*>(&client_addr), 
                                              &addr_len);
        
        if (client_socket == SRT_INVALID_SOCK) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) {
                continue;  // Timeout, check g_running and continue
            }
            if (g_running) {
                std::cerr << "[SRT] Accept error: " << srt_getlasterror_str() << std::endl;
            }
            continue;
        }
        
        char client_ip[INET6_ADDRSTRLEN] = {0};
        int client_port = 0;
        
        if (client_addr.ss_family == AF_INET) {
            sockaddr_in* addr_in = reinterpret_cast<sockaddr_in*>(&client_addr);
            inet_ntop(AF_INET, &addr_in->sin_addr, client_ip, sizeof(client_ip));
            client_port = ntohs(addr_in->sin_port);
        } else if (client_addr.ss_family == AF_INET6) {
            sockaddr_in6* addr_in6 = reinterpret_cast<sockaddr_in6*>(&client_addr);
            inet_ntop(AF_INET6, &addr_in6->sin6_addr, client_ip, sizeof(client_ip));
            client_port = ntohs(addr_in6->sin6_port);
        }
        
        client_counter++;
        
        ClientInfo* client_info = new ClientInfo();
        client_info->socket = client_socket;
        client_info->ip = client_ip;
        client_info->port = client_port;
        client_info->client_num = client_counter;
        client_info->active = true;
        client_info->is_sender = false;
        
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            g_clients.push_back(client_info);
        }
        
        client_info->thread = std::thread(handleClient, client_info);
        client_info->thread.detach();
    }
    
    // Shutdown sequence
    std::cout << "\n[Server] Shutting down..." << std::endl;
    
    // Signal threads to stop
    g_shutdown_started = true;
    g_decoded_frames.shutdown();
    g_encoded_packets.shutdown();
    
    // Wait for threads
    if (ai_thread.joinable()) ai_thread.join();
    if (broadcast_thr.joinable()) broadcast_thr.join();
    
    // Close client connections
    {
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (auto* client : g_clients) {
            if (client->active) {
                srt_close(client->socket);
            }
            delete client;
        }
        g_clients.clear();
    }
    
    // Cleanup GStreamer pipelines
    g_decoder.cleanup();
    g_encoder.cleanup();
    
    // Cleanup SRT
    srt_close(listen_socket);
    srt_cleanup();
    
    // Cleanup GStreamer
    gst_deinit();
    
    std::cout << "[Server] Shutdown complete" << std::endl;
    return 0;
}
