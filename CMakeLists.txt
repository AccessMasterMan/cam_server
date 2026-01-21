cmake_minimum_required(VERSION 3.10)
project(fixed_srt_ai_server VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# ============================================================================
# Find Python3
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

message(STATUS "Python3: ${Python3_VERSION}")
message(STATUS "  Include: ${Python3_INCLUDE_DIRS}")
message(STATUS "  NumPy: ${Python3_NumPy_INCLUDE_DIRS}")

# ============================================================================
# Find OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ============================================================================
# Find SRT
# ============================================================================
find_library(SRT_LIBRARY srt)
find_path(SRT_INCLUDE_DIR srt/srt.h)

if(NOT SRT_LIBRARY OR NOT SRT_INCLUDE_DIR)
    message(FATAL_ERROR "SRT library not found")
endif()

message(STATUS "SRT: ${SRT_LIBRARY}")

# ============================================================================
# Find GStreamer
# ============================================================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.14)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0>=1.14)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0>=1.14)

message(STATUS "GStreamer: ${GSTREAMER_VERSION}")

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SRT_INCLUDE_DIR}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

# ============================================================================
# Source Files
# ============================================================================
set(SOURCES
    fixed_srt_ai_server.cpp
    python_bridge.cpp
)

set(HEADERS
    python_bridge.h
)

# ============================================================================
# Executable
# ============================================================================
add_executable(fixed_srt_ai_server ${SOURCES} ${HEADERS})

# ============================================================================
# Link Libraries
# ============================================================================
target_link_libraries(fixed_srt_ai_server
    ${Python3_LIBRARIES}
    ${OpenCV_LIBS}
    ${SRT_LIBRARY}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    pthread
)

# ============================================================================
# Copy Python Module
# ============================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/ai_worker.py
    COPYONLY
)

# Also create src directory and copy there
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/src/ai_worker.py
    COPYONLY
)

message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════╗")
message(STATUS "║  Fixed SRT AI Server - Ready to Build    ║")
message(STATUS "╚═══════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Build: make -j$(nproc)")
message(STATUS "Run:   ./fixed_srt_ai_server")
message(STATUS "")