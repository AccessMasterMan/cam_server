// fixed_srt_ai_server.cpp
// SRT AI Face Detection Server - Broadcast Architecture
// ONE PORT: Receives H.264 from sender → AI process → Broadcasts processed H.264 to receivers
// Matches your original broadcast server design!

#include <iostream>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <queue>
#include <srt/srt.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include "python_bridge.h"

using namespace FaceStreaming;

std::atomic<bool> g_running(true);

// ============================================================================
// Client Management (same as your broadcast server)
// ============================================================================

struct ClientInfo {
    SRTSOCKET socket;
    std::string ip;
    int port;
    int client_num;
    bool active;
    bool is_sender;
    std::thread thread;
};

std::vector<ClientInfo*> g_clients;
std::mutex g_clients_mutex;

// ============================================================================
// GStreamer Decoder (H.264 → Raw Frames)
// ============================================================================

GstElement* g_decoder_pipeline = nullptr;
GstAppSrc* g_decoder_appsrc = nullptr;
GstAppSink* g_decoder_appsink = nullptr;

std::mutex g_decoded_frame_mutex;
std::queue<cv::Mat> g_decoded_frames;

// Callback: Receive decoded frame from GStreamer
GstFlowReturn on_decoded_frame(GstAppSink* appsink, gpointer user_data) {
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
    
    // Queue for AI processing
    {
        std::lock_guard<std::mutex> lock(g_decoded_frame_mutex);
        g_decoded_frames.push(frame.clone());
    }
    
    return GST_FLOW_OK;
}

bool setup_decoder() {
    std::cout << "[Decoder] Setting up H.264 decoder pipeline..." << std::endl;
    
    // Pipeline: appsrc → h264parse → avdec_h264 → videoconvert → appsink
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-h264,stream-format=byte-stream ! "
        "h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_decoder_pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Decoder] Failed: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    g_decoder_appsrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(g_decoder_pipeline), "src"));
    g_decoder_appsink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(g_decoder_pipeline), "sink"));
    
    GstAppSinkCallbacks callbacks = {nullptr, nullptr, on_decoded_frame};
    gst_app_sink_set_callbacks(g_decoder_appsink, &callbacks, nullptr, nullptr);
    gst_app_sink_set_max_buffers(g_decoder_appsink, 1);
    gst_app_sink_set_drop(g_decoder_appsink, TRUE);
    
    gst_element_set_state(g_decoder_pipeline, GST_STATE_PLAYING);
    
    std::cout << "[Decoder] ✓ Ready" << std::endl;
    return true;
}

// ============================================================================
// GStreamer Encoder (Raw Frames → H.264)
// ============================================================================

GstElement* g_encoder_pipeline = nullptr;
GstAppSrc* g_encoder_appsrc = nullptr;
GstAppSink* g_encoder_appsink = nullptr;

std::mutex g_encoded_packet_mutex;
std::queue<std::vector<uint8_t>> g_encoded_packets;

// Callback: Receive encoded H.264 packets
GstFlowReturn on_encoded_packet(GstAppSink* appsink, gpointer user_data) {
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    
    std::vector<uint8_t> packet(map.data, map.data + map.size);
    
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    
    // Queue for broadcasting
    {
        std::lock_guard<std::mutex> lock(g_encoded_packet_mutex);
        g_encoded_packets.push(packet);
    }
    
    return GST_FLOW_OK;
}

bool setup_encoder(int width, int height, int fps) {
    std::cout << "[Encoder] Setting up H.264 encoder pipeline..." << std::endl;
    
    // Pipeline: appsrc → x264enc → h264parse → mpegtsmux → appsink
    std::string pipeline_str = 
        "appsrc name=src format=time is-live=true do-timestamp=true "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(width) + 
        ",height=" + std::to_string(height) + ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! "
        "video/x-h264,profile=high ! mpegtsmux ! "
        "appsink name=sink emit-signals=true sync=false";
    
    GError* error = nullptr;
    g_encoder_pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Encoder] Failed: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    g_encoder_appsrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(g_encoder_pipeline), "src"));
    g_encoder_appsink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(g_encoder_pipeline), "sink"));
    
    GstAppSinkCallbacks callbacks = {nullptr, nullptr, on_encoded_packet};
    gst_app_sink_set_callbacks(g_encoder_appsink, &callbacks, nullptr, nullptr);
    
    gst_element_set_state(g_encoder_pipeline, GST_STATE_PLAYING);
    
    std::cout << "[Encoder] ✓ Ready" << std::endl;
    return true;
}

// ============================================================================
// AI Processing Thread
// ============================================================================

void ai_processing_thread(PythonBridge& py_bridge) {
    std::cout << "[AI Thread] Started" << std::endl;
    
    std::vector<DetectionResult> detections;
    int frames_processed = 0;
    auto last_stat = std::chrono::steady_clock::now();
    
    while (g_running) {
        cv::Mat frame;
        
        // Get decoded frame
        {
            std::lock_guard<std::mutex> lock(g_decoded_frame_mutex);
            if (g_decoded_frames.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            frame = g_decoded_frames.front();
            g_decoded_frames.pop();
        }
        
        if (frames_processed == 0) {
            std::cout << "[AI Thread] ✓ First frame received!" << std::endl;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // AI detection
        detections.clear();
        if (!py_bridge.detectFaces(frame, detections)) {
            std::cerr << "[AI] Detection failed" << std::endl;
            continue;
        }
        
        // Draw results
        drawDetections(frame, detections);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        frames_processed++;
        
        if (frames_processed <= 3) {
            std::cout << "[AI] Frame #" << frames_processed << " | " 
                      << detections.size() << " face(s) | " << latency_ms << " ms" << std::endl;
        }
        
        // Stats every 5s
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stat).count() >= 5) {
            std::cout << "[AI] Processed " << frames_processed << " frames" << std::endl;
            last_stat = now;
            frames_processed = 0;
        }
        
        // Push to encoder
        if (g_encoder_appsrc) {
            GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame.total() * frame.elemSize(), nullptr);
            GstMapInfo map;
            gst_buffer_map(buffer, &map, GST_MAP_WRITE);
            memcpy(map.data, frame.data, frame.total() * frame.elemSize());
            gst_buffer_unmap(buffer, &map);
            gst_app_src_push_buffer(g_encoder_appsrc, buffer);
        }
    }
    
    std::cout << "[AI Thread] Stopped" << std::endl;
}

// ============================================================================
// Broadcast Thread (same as your original server)
// ============================================================================

void broadcast_thread() {
    std::cout << "[Broadcast Thread] Started" << std::endl;
    
    int packets_sent = 0;
    
    while (g_running) {
        std::vector<uint8_t> packet;
        
        // Get encoded packet
        {
            std::lock_guard<std::mutex> lock(g_encoded_packet_mutex);
            if (g_encoded_packets.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            packet = g_encoded_packets.front();
            g_encoded_packets.pop();
        }
        
        // Broadcast to all receiver clients
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (auto* client : g_clients) {
            if (!client->active || client->is_sender) continue;
            
            int sent = srt_send(client->socket, (const char*)packet.data(), packet.size());
            if (sent == SRT_ERROR) {
                std::cerr << "[Broadcast] Failed to client #" << client->client_num << std::endl;
            }
        }
        
        packets_sent++;
        if (packets_sent % 100 == 1) {
            std::cout << "[Broadcast] Sent " << packets_sent << " packets" << std::endl;
        }
    }
    
    std::cout << "[Broadcast Thread] Stopped" << std::endl;
}

// ============================================================================
// Client Handler (same as your broadcast server)
// ============================================================================

void handleClient(SRTSOCKET client_socket, const char* client_ip, int client_port, int client_num) {
    std::cout << "\n[Client #" << client_num << "] Connected: " << client_ip << ":" << client_port << std::endl;
    
    const int BUFFER_SIZE = 65536;
    uint8_t buffer[BUFFER_SIZE];
    
    int recv_timeout_ms = 100;
    srt_setsockopt(client_socket, 0, SRTO_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
    
    bool is_sender = false;
    bool role_determined = false;
    int chunks_received = 0;
    
    auto last_data = std::chrono::steady_clock::now();
    
    while (g_running) {
        int received = srt_recv(client_socket, (char*)buffer, BUFFER_SIZE);
        
        if (received == SRT_ERROR) {
            int err = srt_getlasterror(nullptr);
            if (err == SRT_EASYNCRCV || err == SRT_ETIMEOUT) {
                auto now = std::chrono::steady_clock::now();
                auto idle = std::chrono::duration_cast<std::chrono::seconds>(now - last_data);
                if (idle.count() >= 15 && chunks_received == 0) {
                    std::cout << "[Client #" << client_num << "] No data - likely receiver" << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            std::cerr << "[Client #" << client_num << "] Error: " << srt_getlasterror_str() << std::endl;
            break;
        }
        
        if (received > 0) {
            chunks_received++;
            last_data = std::chrono::steady_clock::now();
            
            if (!role_determined) {
                is_sender = true;
                role_determined = true;
                std::cout << "[Client #" << client_num << "] ROLE: SENDER" << std::endl;
                
                // Update client info
                std::lock_guard<std::mutex> lock(g_clients_mutex);
                for (auto* c : g_clients) {
                    if (c->socket == client_socket) {
                        c->is_sender = true;
                        break;
                    }
                }
            }
            
            // Push H.264 data to decoder
            if (g_decoder_appsrc) {
                GstBuffer* buf = gst_buffer_new_allocate(nullptr, received, nullptr);
                GstMapInfo map;
                gst_buffer_map(buf, &map, GST_MAP_WRITE);
                memcpy(map.data, buffer, received);
                gst_buffer_unmap(buf, &map);
                gst_app_src_push_buffer(g_decoder_appsrc, buf);
            }
            
            if (chunks_received <= 3) {
                std::cout << "[Client #" << client_num << "] Chunk #" << chunks_received 
                          << " received (" << received << " bytes)" << std::endl;
            }
        }
    }
    
    std::cout << "[Client #" << client_num << "] Disconnected" << std::endl;
    srt_close(client_socket);
    
    std::lock_guard<std::mutex> lock(g_clients_mutex);
    for (auto* client : g_clients) {
        if (client->socket == client_socket) {
            client->active = false;
            break;
        }
    }
}

// ============================================================================
// Signal Handler
// ============================================================================

void signalHandler(int signum) {
    std::cout << "\n[Server] Shutdown signal" << std::endl;
    g_running = false;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SRT AI BROADCAST SERVER                              ║" << std::endl;
    std::cout << "║  Architecture: Sender → AI Processing → Receivers     ║" << std::endl;
    std::cout << "║  ONE PORT (like your original broadcast server!)      ║" << std::endl;
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
    
    // Initialize Python AI
    PythonBridge py_bridge;
    if (!py_bridge.initialize(640)) {
        std::cerr << "[AI] Init failed" << std::endl;
        return 1;
    }
    std::cout << "[AI] ✓ Ready" << std::endl;
    
    // Setup decoder/encoder
    if (!setup_decoder() || !setup_encoder(640, 640, 30)) {
        std::cerr << "[Server] Failed to setup GStreamer" << std::endl;
        return 1;
    }
    
    // Start processing threads
    std::thread ai_thread(ai_processing_thread, std::ref(py_bridge));
    std::thread broadcast_thr(broadcast_thread);
    
    // Create SRT listening socket (SAME as your broadcast server)
    SRTSOCKET listen_socket = srt_create_socket();
    
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
    
    if (srt_bind(listen_socket, (sockaddr*)&sa, sizeof(sa)) == SRT_ERROR) {
        std::cerr << "[SRT] Bind failed" << std::endl;
        return 1;
    }
    
    if (srt_listen(listen_socket, 5) == SRT_ERROR) {
        std::cerr << "[SRT] Listen failed" << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ Listening on port " << PORT << std::endl;
    std::cout << "✓ LIVE mode, latency " << latency_ms << " ms" << std::endl;
    std::cout << "✓ Waiting for clients...\n" << std::endl;
    
    int client_counter = 0;
    
    // Accept clients (SAME as broadcast server)
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
        
        char client_ip[INET6_ADDRSTRLEN];
        int client_port = 0;
        
        if (client_addr.ss_family == AF_INET) {
            sockaddr_in* addr_in = (sockaddr_in*)&client_addr;
            inet_ntop(AF_INET, &addr_in->sin_addr, client_ip, sizeof(client_ip));
            client_port = ntohs(addr_in->sin_port);
        }
        
        client_counter++;
        
        ClientInfo* client_info = new ClientInfo{
            client_socket,
            std::string(client_ip),
            client_port,
            client_counter,
            true,
            false,
            std::thread()
        };
        
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            g_clients.push_back(client_info);
        }
        
        client_info->thread = std::thread(handleClient, client_socket, client_ip, client_port, client_counter);
        client_info->thread.detach();
    }
    
    std::cout << "\n[Server] Shutting down..." << std::endl;
    
    ai_thread.join();
    broadcast_thr.join();
    
    {
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (auto* client : g_clients) {
            if (client->active) srt_close(client->socket);
            delete client;
        }
    }
    
    if (g_decoder_pipeline) {
        gst_element_set_state(g_decoder_pipeline, GST_STATE_NULL);
        gst_object_unref(g_decoder_pipeline);
    }
    
    if (g_encoder_pipeline) {
        gst_element_set_state(g_encoder_pipeline, GST_STATE_NULL);
        gst_object_unref(g_encoder_pipeline);
    }
    
    srt_close(listen_socket);
    srt_cleanup();
    gst_deinit();
    
    std::cout << "[Server] Shutdown complete" << std::endl;
    return 0;
}