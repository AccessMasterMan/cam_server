// src/srt_ai_server.cpp
// SRT AI Face Detection Server
// Single-client optimized with embedded Python (zero-copy)

#include <iostream>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <srt/srt.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "python_bridge.h"

using namespace FaceStreaming;

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running(true);
std::atomic<bool> g_ai_ready(false);

enum class ServerState {
    INITIALIZING,   // Python loading
    WARMING_UP,     // TensorRT warmup
    READY,          // Ready to accept connections
    PROCESSING,     // Client connected, processing frames
    ERROR           // Fatal error
};

std::atomic<ServerState> g_server_state(ServerState::INITIALIZING);

// ============================================================================
// Signal Handler
// ============================================================================

void signalHandler(int signum) {
    std::cout << "\n[Server] Shutdown signal received" << std::endl;
    g_running = false;
}

// ============================================================================
// SRT Frame Receiver (Optimized)
// ============================================================================

class SRTFrameReceiver {
public:
    SRTFrameReceiver() : width_(640), height_(640) {
        // Pre-allocate frame buffer (reused for all frames)
        frame_buffer_.create(height_, width_, CV_8UC3);
    }
    
    bool receiveFrame(SRTSOCKET socket, cv::Mat& frame) {
        // Expected frame size (640x640x3 = 1,228,800 bytes)
        const int expected_size = width_ * height_ * 3;
        
        // Receive data into pre-allocated buffer
        int received = srt_recv(socket, (char*)frame_buffer_.data, expected_size);
        
        if (received == SRT_ERROR) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) {
                return false;  // No data available
            }
            std::cerr << "[SRT] Receive error: " << srt_getlasterror_str() << std::endl;
            return false;
        }
        
        if (received != expected_size) {
            std::cerr << "[SRT] Incomplete frame: got " << received 
                      << " bytes, expected " << expected_size << std::endl;
            return false;
        }
        
        // Return reference to our internal buffer (avoid copy)
        frame = frame_buffer_;
        return true;
    }
    
private:
    int width_;
    int height_;
    cv::Mat frame_buffer_;  // Reused buffer
};

// ============================================================================
// SRT Frame Sender (Optimized)
// ============================================================================

class SRTFrameSender {
public:
    bool sendFrame(SRTSOCKET socket, const cv::Mat& frame) {
        if (!frame.isContinuous()) {
            std::cerr << "[SRT] Frame must be continuous for sending" << std::endl;
            return false;
        }
        
        int size = frame.total() * frame.elemSize();
        
        int sent = srt_send(socket, (const char*)frame.data, size);
        
        if (sent == SRT_ERROR) {
            std::cerr << "[SRT] Send error: " << srt_getlasterror_str() << std::endl;
            return false;
        }
        
        if (sent != size) {
            std::cerr << "[SRT] Incomplete send: " << sent << "/" << size << std::endl;
            return false;
        }
        
        return true;
    }
};

// ============================================================================
// Performance Monitor
// ============================================================================

class PerformanceMonitor {
public:
    PerformanceMonitor() : frame_count_(0), total_latency_ms_(0.0) {
        last_report_ = std::chrono::steady_clock::now();
    }
    
    void recordFrame(double latency_ms) {
        frame_count_++;
        total_latency_ms_ += latency_ms;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_);
        
        if (elapsed.count() >= 5) {  // Report every 5 seconds
            double avg_latency = total_latency_ms_ / frame_count_;
            double fps = frame_count_ / elapsed.count();
            
            std::cout << "\n[Performance] " << frame_count_ << " frames in " 
                      << elapsed.count() << "s" << std::endl;
            std::cout << "              Avg latency: " << avg_latency << " ms" << std::endl;
            std::cout << "              FPS: " << fps << std::endl;
            std::cout << "              Throughput: " << (1000.0 / avg_latency) << " max FPS\n" << std::endl;
            
            // Reset counters
            frame_count_ = 0;
            total_latency_ms_ = 0.0;
            last_report_ = now;
        }
    }
    
private:
    uint64_t frame_count_;
    double total_latency_ms_;
    std::chrono::steady_clock::time_point last_report_;
};

// ============================================================================
// Client Handler
// ============================================================================

void handleClient(SRTSOCKET client_socket, const char* client_ip, int client_port,
                 PythonBridge& py_bridge) {
    
    std::cout << "\n╔═══════════════════════════════════════════╗" << std::endl;
    std::cout << "║  CLIENT CONNECTED                         ║" << std::endl;
    std::cout << "║  IP: " << client_ip << ":" << client_port << std::endl;
    std::cout << "╚═══════════════════════════════════════════╝\n" << std::endl;
    
    g_server_state = ServerState::PROCESSING;
    
    // Set receive timeout
    int recv_timeout_ms = 100;
    srt_setsockopt(client_socket, 0, SRTO_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
    
    SRTFrameReceiver receiver;
    SRTFrameSender sender;
    PerformanceMonitor perf_mon;
    
    std::vector<DetectionResult> detections;
    cv::Mat frame;
    
    int frames_processed = 0;
    
    std::cout << "[Server] Waiting for video stream..." << std::endl;
    
    while (g_running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // ===== STEP 1: Receive Frame =====
        if (!receiver.receiveFrame(client_socket, frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        if (frames_processed == 0) {
            std::cout << "\n[Server] ✓ First frame received! Starting AI processing...\n" << std::endl;
        }
        
        auto recv_time = std::chrono::high_resolution_clock::now();
        
        // ===== STEP 2: AI Detection (Embedded Python) =====
        detections.clear();
        if (!py_bridge.detectFaces(frame, detections)) {
            std::cerr << "[Server] AI detection failed: " << py_bridge.getLastError() << std::endl;
            continue;
        }
        
        auto detect_time = std::chrono::high_resolution_clock::now();
        
        // ===== STEP 3: Draw Results (C++ OpenCV) =====
        drawDetections(frame, detections);
        
        auto draw_time = std::chrono::high_resolution_clock::now();
        
        // ===== STEP 4: Send Back to Client =====
        if (!sender.sendFrame(client_socket, frame)) {
            std::cerr << "[Server] Failed to send frame" << std::endl;
            break;
        }
        
        auto send_time = std::chrono::high_resolution_clock::now();
        
        // ===== Performance Tracking =====
        double recv_ms = std::chrono::duration<double, std::milli>(recv_time - frame_start).count();
        double detect_ms = std::chrono::duration<double, std::milli>(detect_time - recv_time).count();
        double draw_ms = std::chrono::duration<double, std::milli>(draw_time - detect_time).count();
        double send_ms = std::chrono::duration<double, std::milli>(send_time - draw_time).count();
        double total_ms = std::chrono::duration<double, std::milli>(send_time - frame_start).count();
        
        frames_processed++;
        perf_mon.recordFrame(total_ms);
        
        // Log first few frames with detailed timing
        if (frames_processed <= 5) {
            std::cout << "[Frame #" << frames_processed << "] "
                      << detections.size() << " face(s) detected | "
                      << "Timings: recv=" << recv_ms << "ms, "
                      << "detect=" << detect_ms << "ms, "
                      << "draw=" << draw_ms << "ms, "
                      << "send=" << send_ms << "ms, "
                      << "TOTAL=" << total_ms << "ms" << std::endl;
        }
    }
    
    std::cout << "\n[Server] Client disconnected" << std::endl;
    std::cout << "[Server] Total frames processed: " << frames_processed << std::endl;
    
    srt_close(client_socket);
    
    g_server_state = ServerState::READY;
}

// ============================================================================
// Main Server
// ============================================================================

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SRT AI FACE DETECTION SERVER                         ║" << std::endl;
    std::cout << "║  Single-Client Optimized | Embedded Python            ║" << std::endl;
    std::cout << "║  Zero-Copy Frame Transfer | TensorRT FP16             ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // ===== INITIALIZATION PHASE =====
    std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  PHASE 1: INITIALIZATION                              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    // Initialize SRT
    if (srt_startup() < 0) {
        std::cerr << "[Server] Failed to initialize SRT: " << srt_getlasterror_str() << std::endl;
        return 1;
    }
    
    std::cout << "[Server] SRT library initialized (v" << srt_getversion() << ")" << std::endl;
    
    // Initialize Python Bridge
    PythonBridge py_bridge;
    
    g_server_state = ServerState::WARMING_UP;
    
    if (!py_bridge.initialize(640)) {
        std::cerr << "[Server] Failed to initialize Python bridge" << std::endl;
        std::cerr << "[Server] Error: " << py_bridge.getLastError() << std::endl;
        g_server_state = ServerState::ERROR;
        srt_cleanup();
        return 1;
    }
    
    g_ai_ready = true;
    g_server_state = ServerState::READY;
    
    // ===== SRT SERVER SETUP =====
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  PHASE 2: SRT SERVER SETUP                            ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    SRTSOCKET listen_socket = srt_create_socket();
    if (listen_socket == SRT_INVALID_SOCK) {
        std::cerr << "[Server] Failed to create socket: " << srt_getlasterror_str() << std::endl;
        srt_cleanup();
        return 1;
    }
    
    // LIVE mode for real-time streaming
    int live_mode = SRTT_LIVE;
    srt_setsockopt(listen_socket, 0, SRTO_TRANSTYPE, &live_mode, sizeof(live_mode));
    
    // Ultra-low latency settings
    int latency_ms = 10;  // 10ms latency
    srt_setsockopt(listen_socket, 0, SRTO_LATENCY, &latency_ms, sizeof(latency_ms));
    
    int tsbpd_mode = 1;
    srt_setsockopt(listen_socket, 0, SRTO_TSBPDMODE, &tsbpd_mode, sizeof(tsbpd_mode));
    
    // Buffer sizes (4MB for low latency)
    int rcvbuf = 4000000;
    int sndbuf = 4000000;
    srt_setsockopt(listen_socket, 0, SRTO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    srt_setsockopt(listen_socket, 0, SRTO_SNDBUF, &sndbuf, sizeof(sndbuf));
    
    // Bind and listen
    const int PORT = 9000;
    sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(PORT);
    sa.sin_addr.s_addr = INADDR_ANY;
    
    if (srt_bind(listen_socket, (sockaddr*)&sa, sizeof(sa)) == SRT_ERROR) {
        std::cerr << "[Server] Bind failed: " << srt_getlasterror_str() << std::endl;
        srt_close(listen_socket);
        srt_cleanup();
        return 1;
    }
    
    if (srt_listen(listen_socket, 1) == SRT_ERROR) {  // Accept only 1 client
        std::cerr << "[Server] Listen failed: " << srt_getlasterror_str() << std::endl;
        srt_close(listen_socket);
        srt_cleanup();
        return 1;
    }
    
    std::cout << "✓ Listening on port " << PORT << " (LIVE mode)" << std::endl;
    std::cout << "✓ Latency: " << latency_ms << " ms (ultra-low)" << std::endl;
    std::cout << "✓ Single-client mode (optimized)" << std::endl;
    
    // ===== READY TO ACCEPT CONNECTIONS =====
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SERVER READY                                         ║" << std::endl;
    std::cout << "║  ✓ AI Engine: READY                                   ║" << std::endl;
    std::cout << "║  ✓ SRT Server: LISTENING                              ║" << std::endl;
    std::cout << "║  Waiting for client connection...                     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    // ===== ACCEPT LOOP =====
    while (g_running) {
        sockaddr_storage client_addr;
        int addr_len = sizeof(client_addr);
        
        SRTSOCKET client_socket = srt_accept(listen_socket, (sockaddr*)&client_addr, &addr_len);
        
        if (client_socket == SRT_INVALID_SOCK) {
            if (g_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }
        
        // Get client info
        char client_ip[INET6_ADDRSTRLEN];
        int client_port = 0;
        
        if (client_addr.ss_family == AF_INET) {
            sockaddr_in* addr_in = (sockaddr_in*)&client_addr;
            inet_ntop(AF_INET, &addr_in->sin_addr, client_ip, sizeof(client_ip));
            client_port = ntohs(addr_in->sin_port);
        }
        
        // Handle client (blocking - single client design)
        handleClient(client_socket, client_ip, client_port, py_bridge);
        
        std::cout << "\n[Server] Ready to accept new connection..." << std::endl;
    }
    
    // ===== CLEANUP =====
    std::cout << "\n[Server] Shutting down..." << std::endl;
    
    srt_close(listen_socket);
    srt_cleanup();
    
    std::cout << "[Server] Shutdown complete" << std::endl;
    
    return 0;
}

