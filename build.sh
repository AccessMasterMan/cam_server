#!/bin/bash
# Quick build script for SRT AI Server

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  SRT AI Server - Build Script                         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check dependencies
echo -e "${YELLOW}[1/5] Checking dependencies...${NC}"

# Check for Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}ERROR: cmake not found. Install with: sudo apt-get install cmake${NC}"
    exit 1
fi

# Check for SRT library
if ! ldconfig -p | grep -q libsrt; then
    echo -e "${RED}ERROR: libsrt not found. Install with: sudo apt-get install libsrt-dev${NC}"
    exit 1
fi

# Check Python packages
echo -e "${YELLOW}[2/5] Checking Python packages...${NC}"
python3 -c "import numpy, cv2, insightface" 2>/dev/null || {
    echo -e "${RED}ERROR: Missing Python packages${NC}"
    echo "Install with: pip3 install numpy opencv-python insightface onnxruntime-gpu"
    exit 1
}

echo -e "${GREEN}✓ All dependencies found${NC}"

# Create build directory
echo -e "${YELLOW}[3/5] Creating build directory...${NC}"
mkdir -p build
cd build

# Run CMake
echo -e "${YELLOW}[4/5] Running CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${YELLOW}[5/5] Building project...${NC}"
make -j$(nproc)

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Build Complete!                                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Executable: ./build/srt_ai_server${NC}"
echo ""
echo "To run the server:"
echo "  cd build"
echo "  ./srt_ai_server"
echo ""
echo "To test Python module standalone:"
echo "  cd build"
echo "  python3 ai_worker.py"
echo ""

