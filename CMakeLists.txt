cmake_minimum_required(VERSION 3.14)
project(srt_ai_server VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# ============================================================================
# Find pybind11 (REQUIRED for embedding Python)
# ============================================================================
# Try to find pybind11 via pip install location first
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    message(STATUS "Found pybind11 via pip: ${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11: ${pybind11_VERSION}")
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIRS}")

# ============================================================================
# Find Python3 (for NumPy headers)
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
    message(FATAL_ERROR "SRT library not found. Install with: apt install libsrt-dev")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
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
    src/srt_ai_server.cpp
    src/python_bridge.cpp
)

set(HEADERS
    src/python_bridge.h
)

# ============================================================================
# Executable
# ============================================================================
add_executable(srt_ai_server ${SOURCES} ${HEADERS})

# ============================================================================
# Link Libraries
# CRITICAL: Use pybind11::embed for embedding Python interpreter
# ============================================================================
target_link_libraries(srt_ai_server PRIVATE
    pybind11::embed
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/ai_worker.py
    COPYONLY
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/src/ai_worker.py
    COPYONLY
)

# Create TensorRT cache directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/trt_cache)

# ============================================================================
# Print Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════╗")
message(STATUS "║  SRT AI Server (pybind11) - Build Configuration           ║")
message(STATUS "╚═══════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  pybind11:   ${pybind11_VERSION}")
message(STATUS "  Python:     ${Python3_VERSION}")
message(STATUS "  OpenCV:     ${OpenCV_VERSION}")
message(STATUS "  GStreamer:  ${GSTREAMER_VERSION}")
message(STATUS "  SRT:        Found")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  ./srt_ai_server")
message(STATUS "")