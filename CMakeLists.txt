cmake_minimum_required(VERSION 3.10)
project(srt_ai_server VERSION 1.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build Type (Release by default for performance)
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Compiler Flags
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# ============================================================================
# Find Python3
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

message(STATUS "Python3 found:")
message(STATUS "  Version: ${Python3_VERSION}")
message(STATUS "  Include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${Python3_LIBRARIES}")
message(STATUS "  NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")

# ============================================================================
# Find OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV found:")
message(STATUS "  Version: ${OpenCV_VERSION}")
message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${OpenCV_LIBS}")

# ============================================================================
# Find SRT
# ============================================================================
find_library(SRT_LIBRARY srt)
find_path(SRT_INCLUDE_DIR srt/srt.h)

if(NOT SRT_LIBRARY OR NOT SRT_INCLUDE_DIR)
    message(FATAL_ERROR "SRT library not found. Install with: sudo apt-get install libsrt-dev")
endif()

message(STATUS "SRT found:")
message(STATUS "  Library: ${SRT_LIBRARY}")
message(STATUS "  Include dir: ${SRT_INCLUDE_DIR}")

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SRT_INCLUDE_DIR}
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
# ============================================================================
target_link_libraries(srt_ai_server
    ${Python3_LIBRARIES}
    ${OpenCV_LIBS}
    ${SRT_LIBRARY}
    pthread
)

# ============================================================================
# Python Module (ai_worker.py) - Copy to build directory
# ============================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/ai_worker.py
    COPYONLY
)

# Also create symlink in src directory for development
add_custom_command(TARGET srt_ai_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_worker.py
    ${CMAKE_CURRENT_BINARY_DIR}/src/ai_worker.py
)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS srt_ai_server DESTINATION bin)
install(FILES src/ai_worker.py DESTINATION bin)

# ============================================================================
# Display Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════╗")
message(STATUS "║  SRT AI Server - Build Configuration                  ║")
message(STATUS "╚═══════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python version:   ${Python3_VERSION}")
message(STATUS "  OpenCV version:   ${OpenCV_VERSION}")
message(STATUS "  SRT library:      ${SRT_LIBRARY}")
message(STATUS "")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build directory:  ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════╗")
message(STATUS "║  Build Commands:                                      ║")
message(STATUS "║    mkdir build && cd build                            ║")
message(STATUS "║    cmake ..                                           ║")
message(STATUS "║    make -j$(nproc)                                    ║")
message(STATUS "║    ./srt_ai_server                                    ║")
message(STATUS "╚═══════════════════════════════════════════════════════╝")
message(STATUS "")
