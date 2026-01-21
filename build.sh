#!/bin/bash
# build.sh - Simple build script for SRT AI Server
# Run this from /workspace/cam_server directory

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  SRT AI Server - Build Script                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "src" ]; then
    echo "ERROR: Please run this script from /workspace/cam_server directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "[1/3] Cleaning old build..."
if [ -d "build" ]; then
    rm -rf build
    echo "✓ Old build removed"
fi
mkdir -p build
echo "✓ Fresh build directory created"

echo ""
echo "[2/3] Configuring with CMake..."
cd build
if ! cmake ..; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    exit 1
fi
echo "✓ Configuration successful"

echo ""
echo "[3/3] Compiling..."
if ! make -j$(nproc); then
    echo ""
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "✓ Compilation successful"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  BUILD SUCCESSFUL!                                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Executable: $(pwd)/srt_ai_server"
echo ""
echo "To run the server:"
echo "  ./srt_ai_server"
echo ""
echo "Or from project root:"
echo "  cd build && ./srt_ai_server"
echo ""

# Verify ai_worker.py exists
if [ ! -f "ai_worker.py" ] && [ -f "../src/ai_worker.py" ]; then
    echo "Note: Copying ai_worker.py to build directory..."
    cp ../src/ai_worker.py .
    mkdir -p src
    cp ../src/ai_worker.py src/
fi