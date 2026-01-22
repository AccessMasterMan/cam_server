// srt_ai_server.cpp - SRT AI Face Detection Server with pybind11
// 
// Architecture: Sender → AI Processing → Receivers (Broadcast)
// Uses pybind11 for safe Python embedding with proper GIL management
//
// Key design:
// - py::scoped_interpreter in main() owns the Python lifetime
// - GIL released after initialization for multi-threaded access
// - Worker threads use py::gil_scoped_acquire when calling Python

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

// pybind11 MUST be included before Python.h (which it includes internally)
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
// Thread-Safe Queue
// ============================================================================

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> shutdown_;
    
public:
    explicit ThreadSafeQueue(size_t max_size = 10) 
        : max_size_(max_size), shutdown_(false) {}
    
    ~ThreadSafeQueue() { shutdown(); }
    
    void shutdown() {
        shutdown_ = true;
        cv_.notify_all();
    }
    
    bool push_drop_oldest(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) return false;
        
        // Drop oldest if full (for low-latency)
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
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// Global queues
ThreadSafeQueue<cv::Mat> g_decoded_frames(5);
ThreadSafeQueue<std::vector<uint8_t>> g_encoded_packets(30);

// ============================================================================
// GStreamer Pipeline Wrapper
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
            GstState state;
            gst_element_get_state(pipeline, &state, nullptr, GST_SECOND);
        }
        
        if (appsrc) { gst_object_unref(appsrc); appsrc = nullptr; }
        if (appsink) { gst_object_unref(appsink); appsink = nullptr; }
        if (pipeline) { gst_object_unref(pipeline); pipeline = nullptr; }
    }
};

GstPipelineWrapper g_decoder;
GstPipelineWrapper g_encoder;

// ============================================================================
// GStreamer Callbacks
// ============================================================================

GstFlowReturn on_decoded_frame(GstAppSink* appsink, gpointer) {
    if (!g_running || g_shutdown_started) return GST_FLOW_EOS;
    
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    
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
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    
    cv::Mat frame(height, width, CV_8UC3);
    memcpy(frame.data, map.data, expected_size);
    
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
    
    g_encoded_packets.push_drop_oldest(packet);
    return GST_FLOW_OK;
}

// ============================================================================
// Pipeline Setup
// ============================================================================

bool setup_decoder() {
    std::cout << "\n[Decoder] Setting up H.264 decoder pipeline..." << std::endl;
    
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-h264,stream-format=byte-stream ! "
        "h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_decoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Decoder] Failed: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    g_decoder.appsrc = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "src");
    g_decoder.appsink = gst_bin_get_by_name(GST_BIN(g_decoder.pipeline), "sink");
    
    if (!g_decoder.appsrc || !g_decoder.appsink) {
        std::cerr << "[Decoder] Failed to get elements" << std::endl;
        return false;
    }
    
    g_object_set(G_OBJECT(g_decoder.appsrc),
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);
    
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_decoded_frame;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_decoder.appsink), &callbacks, nullptr, nullptr);
    gst_app_sink_set_max_buffers(GST_APP_SINK(g_decoder.appsink), 1);
    gst_app_sink_set_drop(GST_APP_SINK(g_decoder.appsink), TRUE);
    
    GstStateChangeReturn ret = gst_element_set_state(g_decoder.pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Decoder] Failed to start" << std::endl;
        return false;
    }
    
    GstState state;
    gst_element_get_state(g_decoder.pipeline, &state, nullptr, 2 * GST_SECOND);
    
    std::cout << "[Decoder] ✓ Ready" << std::endl;
    return true;
}

bool setup_encoder(int width, int height, int fps) {
    std::cout << "[Encoder] Setting up H.264 encoder pipeline..." << std::endl;
    
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(width) + 
        ",height=" + std::to_string(height) + ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! "
        "video/x-h264,profile=high ! mpegtsmux ! "
        "appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_encoder.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Encoder] Failed: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    g_encoder.appsrc = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "src");
    g_encoder.appsink = gst_bin_get_by_name(GST_BIN(g_encoder.pipeline), "sink");
    
    if (!g_encoder.appsrc || !g_encoder.appsink) {
        std::cerr << "[Encoder] Failed to get elements" << std::endl;
        return false;
    }
    
    g_object_set(G_OBJECT(g_encoder.appsrc),
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);
    
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_encoded_packet;
    gst_app_sink_set_callbacks(GST_APP_SINK(g_encoder.appsink), &callbacks, nullptr, nullptr);
    
    GstStateChangeReturn ret = gst_element_set_state(g_encoder.pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Encoder] Failed to start" << std::endl;
        return false;
    }
    
    GstState state;
    gst_element_get_state(g_encoder.pipeline, &state, nullptr, 2 * GST_SECOND);
    
    std::cout << "[Encoder] ✓ Ready\n" << std::endl;
    return true;
}

// ============================================================================
// Thread-Safe Pipeline Access
// ============================================================================

bool push_frame_to_encoder(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(g_encoder.mutex);
    if (!g_encoder.appsrc || !g_running) return false;
    
    size_t frame_size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
    if (!buffer) return false;
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return false;
    }
    
    memcpy(map.data, frame.data, frame_size);
    gst_buffer_unmap(buffer, &map);
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(g_encoder.appsrc), buffer);
    return (ret == GST_FLOW_OK);
}

bool push_data_to_decoder(const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(g_decoder.mutex);
    if (!g_decoder.appsrc || !g_running) return false;
    
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    if (!buffer) return false;
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return false;
    }
    
    memcpy(map.data, data, size);
    gst_buffer_unmap(buffer, &map);
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(g_decoder.appsrc), buffer);
    return (ret == GST_FLOW_OK);
}

// ============================================================================
// AI Processing Thread
// ============================================================================

void ai_processing_thread(PythonBridge& py_bridge) {
    std::cout << "[AI Thread] Started" << std::endl;
    
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
        
        if (!g_decoded_frames.pop(frame, 100)) {
            continue;
        }
        
        if (frame.empty()) continue;
        
        if (frames_processed == 0) {
            std::cout << "[AI Thread] ✓ First frame: " 
                      << frame.cols << "x" << frame.rows << std::endl;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        detections.clear();
        if (!py_bridge.detectFaces(frame, detections)) {
            std::cerr << "[AI] Detection failed: " << py_bridge.getLastError() << std::endl;
            continue;
        }
        
        drawDetections(frame, detections);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        frames_processed++;
        
        if (frames_processed <= 3) {
            std::cout << "[AI] Frame #" << frames_processed << " | " 
                      << detections.size() << " face(s) | " << latency_ms << " ms" << std::endl;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stat).count() >= 5) {
            std::cout << "[AI] Processed " << frames_processed << " frames in 5s" << std::endl;
            last_stat = now;
            frames_processed = 0;
        }
        
        push_frame_to_encoder(frame);
    }
    
    std::cout << "[AI Thread] Stopped" << std::endl;
}

// ============================================================================
// Broadcast Thread
// ============================================================================

void broadcast_thread() {
    std::cout << "[Broadcast Thread] Started" << std::endl;
    
    while (!g_pipelines_ready && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!g_running) {
        std::cout << "[Broadcast Thread] Stopped" << std::endl;
        return;
    }
    
    std::cout << "[Broadcast Thread] Ready" << std::endl;
    
    int packets_sent = 0;
    
    while (g_running && !g_shutdown_started) {
        std::vector<uint8_t> packet;
        
        if (!g_encoded_packets.pop(packet, 100)) continue;
        if (packet.empty()) continue;
        
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            for (auto* client : g_clients) {
                if (!client->active || client->is_sender) continue;
                
                int sent = srt_send(client->socket, 
                                    reinterpret_cast<const char*>(packet.data()), 
                                    static_cast<int>(packet.size()));
                if (sent == SRT_ERROR) {
                    client->active = false;
                }
            }
        }
        
        packets_sent++;
        if (packets_sent == 1 || packets_sent % 100 == 0) {
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
    
    int recv_timeout_ms = 100;
    srt_setsockopt(client_socket, 0, SRTO_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
    
    bool role_determined = false;
    int chunks_received = 0;
    auto last_data = std::chrono::steady_clock::now();
    
    while (g_running && client_info->active && !g_shutdown_started) {
        int received = srt_recv(client_socket, 
                                reinterpret_cast<char*>(buffer.data()), 
                                BUFFER_SIZE);
        
        if (received == SRT_ERROR) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) {
                auto now = std::chrono::steady_clock::now();
                auto idle = std::chrono::duration_cast<std::chrono::seconds>(now - last_data);
                if (idle.count() >= 10 && chunks_received == 0 && !role_determined) {
                    std::cout << "[Client #" << client_num << "] ROLE: RECEIVER" << std::endl;
                    role_determined = true;
                    client_info->is_sender = false;
                }
                continue;
            }
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
            
            push_data_to_decoder(buffer.data(), received);
            
            if (chunks_received <= 3) {
                std::cout << "[Client #" << client_num << "] Chunk #" << chunks_received 
                          << " (" << received << " bytes)" << std::endl;
            }
        }
    }
    
    std::cout << "[Client #" << client_num << "] Disconnected" << std::endl;
    srt_close(client_socket);
    client_info->active = false;
}

// ============================================================================
// Signal Handler
// ============================================================================

void signalHandler(int) {
    std::cout << "\n[Server] Shutdown signal" << std::endl;
    g_running = false;
    g_shutdown_started = true;
    g_decoded_frames.shutdown();
    g_encoded_packets.shutdown();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SRT AI BROADCAST SERVER (pybind11 edition)           ║" << std::endl;
    std::cout << "║  Architecture: Sender → AI Processing → Receivers     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Initialize GStreamer
    gst_init(nullptr, nullptr);
    std::cout << "[GStreamer] Initialized" << std::endl;
    
    // Initialize SRT
    if (srt_startup() < 0) {
        std::cerr << "[SRT] Init failed" << std::endl;
        return 1;
    }
    std::cout << "[SRT] Initialized" << std::endl;
    
    // ========================================================================
    // CRITICAL: Python interpreter lifetime managed by scoped_interpreter
    // This MUST be in main() and stay alive for the entire program
    // ========================================================================
    py::scoped_interpreter python_guard{};
    std::cout << "[Python] Interpreter started (pybind11)" << std::endl;
    
    // Initialize the Python bridge (loads AI model)
    PythonBridge py_bridge;
    if (!py_bridge.initialize(640)) {
        std::cerr << "[AI] Init failed: " << py_bridge.getLastError() << std::endl;
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    std::cout << "[AI] ✓ Ready\n" << std::endl;
    
    // Setup GStreamer pipelines
    if (!setup_decoder()) {
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    if (!setup_encoder(640, 640, 30)) {
        g_decoder.cleanup();
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    // ========================================================================
    // CRITICAL: Release GIL BEFORE starting threads
    // This allows worker threads to acquire the GIL
    // ========================================================================
    py_bridge.releaseGIL();
    
    // Allow pipelines to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    g_pipelines_ready = true;
    std::cout << "[Server] ✓ Pipelines ready" << std::endl;
    
    // Start processing threads
    std::cout << "[Server] Starting threads..." << std::endl;
    std::thread ai_thread(ai_processing_thread, std::ref(py_bridge));
    std::thread broadcast_thr(broadcast_thread);
    std::cout << "[Server] ✓ Threads started\n" << std::endl;
    
    // Create SRT socket
    SRTSOCKET listen_socket = srt_create_socket();
    
    int live_mode = SRTT_LIVE;
    srt_setsockopt(listen_socket, 0, SRTO_TRANSTYPE, &live_mode, sizeof(live_mode));
    
    int latency_ms = 10;
    srt_setsockopt(listen_socket, 0, SRTO_LATENCY, &latency_ms, sizeof(latency_ms));
    
    int tsbpd_mode = 1;
    srt_setsockopt(listen_socket, 0, SRTO_TSBPDMODE, &tsbpd_mode, sizeof(tsbpd_mode));
    
    int bufsize = 4000000;
    srt_setsockopt(listen_socket, 0, SRTO_RCVBUF, &bufsize, sizeof(bufsize));
    srt_setsockopt(listen_socket, 0, SRTO_SNDBUF, &bufsize, sizeof(bufsize));
    
    const int PORT = 9000;
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(PORT);
    sa.sin_addr.s_addr = INADDR_ANY;
    
    if (srt_bind(listen_socket, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == SRT_ERROR) {
        std::cerr << "[SRT] Bind failed" << std::endl;
        g_running = false;
        g_decoded_frames.shutdown();
        g_encoded_packets.shutdown();
        ai_thread.join();
        broadcast_thr.join();
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    if (srt_listen(listen_socket, 5) == SRT_ERROR) {
        std::cerr << "[SRT] Listen failed" << std::endl;
        g_running = false;
        g_decoded_frames.shutdown();
        g_encoded_packets.shutdown();
        ai_thread.join();
        broadcast_thr.join();
        srt_close(listen_socket);
        srt_cleanup();
        gst_deinit();
        return 1;
    }
    
    std::cout << "════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  ✓ Listening on port " << PORT << std::endl;
    std::cout << "  ✓ LIVE mode, latency " << latency_ms << " ms" << std::endl;
    std::cout << "  ✓ Waiting for clients..." << std::endl;
    std::cout << "════════════════════════════════════════════════════════\n" << std::endl;
    
    int client_counter = 0;
    int accept_timeout = 500;
    srt_setsockopt(listen_socket, 0, SRTO_RCVTIMEO, &accept_timeout, sizeof(accept_timeout));
    
    // Accept loop
    while (g_running) {
        sockaddr_storage client_addr;
        int addr_len = sizeof(client_addr);
        
        SRTSOCKET client_socket = srt_accept(listen_socket, 
                                              reinterpret_cast<sockaddr*>(&client_addr), 
                                              &addr_len);
        
        if (client_socket == SRT_INVALID_SOCK) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) continue;
            if (g_running) {
                std::cerr << "[SRT] Accept error" << std::endl;
            }
            continue;
        }
        
        char client_ip[INET6_ADDRSTRLEN] = {0};
        int client_port = 0;
        
        if (client_addr.ss_family == AF_INET) {
            auto* addr_in = reinterpret_cast<sockaddr_in*>(&client_addr);
            inet_ntop(AF_INET, &addr_in->sin_addr, client_ip, sizeof(client_ip));
            client_port = ntohs(addr_in->sin_port);
        }
        
        client_counter++;
        
        auto* client_info = new ClientInfo();
        client_info->socket = client_socket;
        client_info->ip = client_ip;
        client_info->port = client_port;
        client_info->client_num = client_counter;
        client_info->active = true;
        
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            g_clients.push_back(client_info);
        }
        
        client_info->thread = std::thread(handleClient, client_info);
        client_info->thread.detach();
    }
    
    // Shutdown
    std::cout << "\n[Server] Shutting down..." << std::endl;
    
    g_shutdown_started = true;
    g_decoded_frames.shutdown();
    g_encoded_packets.shutdown();
    
    if (ai_thread.joinable()) ai_thread.join();
    if (broadcast_thr.joinable()) broadcast_thr.join();
    
    {
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (auto* client : g_clients) {
            if (client->active) srt_close(client->socket);
            delete client;
        }
        g_clients.clear();
    }
    
    g_decoder.cleanup();
    g_encoder.cleanup();
    
    srt_close(listen_socket);
    srt_cleanup();
    gst_deinit();
    
    std::cout << "[Server] Shutdown complete" << std::endl;
    return 0;
}