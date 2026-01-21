#!/bin/bash
# install_requirements.sh
# System-level dependency installer for SRT AI Server
# Run this BEFORE building the project in Docker containers

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  SRT AI Server - System Requirements Installer        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}ERROR: This script must be run as root${NC}"
    echo "Usage: sudo ./install_requirements.sh"
    exit 1
fi

echo -e "${YELLOW}[1/6] Updating package lists...${NC}"
apt-get update || {
    echo -e "${RED}Failed to update package lists${NC}"
    exit 1
}

echo -e "${GREEN}✓ Package lists updated${NC}"
echo ""

# ============================================================================
# OpenCV Dependencies (Critical for opencv-python from pip)
# ============================================================================
echo -e "${YELLOW}[2/6] Installing OpenCV system dependencies...${NC}"
echo "These are required for opencv-python to work properly:"

OPENCV_DEPS=(
    libgl1              # OpenGL library (required by OpenCV GUI)
    libglib2.0-0        # GLib library (required by OpenCV)
    libsm6              # X11 Session Management library
    libxext6            # X11 extensions library
    libxrender1         # X11 Render extension
    libgomp1            # GNU OpenMP library (for parallel processing)
    libavcodec-dev      # FFmpeg video codec library
    libavformat-dev     # FFmpeg format library
    libswscale-dev      # FFmpeg scaling library
)

apt-get install -y libopencv-dev

apt-get install -y "${OPENCV_DEPS[@]}" || {
    echo -e "${RED}Failed to install OpenCV dependencies${NC}"
    exit 1
}

echo -e "${GREEN}✓ OpenCV dependencies installed${NC}"
echo ""

# ============================================================================
# Build Tools
# ============================================================================
echo -e "${YELLOW}[3/6] Installing build tools...${NC}"

BUILD_TOOLS=(
    build-essential     # GCC, G++, make
    cmake               # CMake build system
    pkg-config          # Package configuration tool
    git                 # Version control (optional, for development)
)

apt-get install -y "${BUILD_TOOLS[@]}" || {
    echo -e "${RED}Failed to install build tools${NC}"
    exit 1
}

echo -e "${GREEN}✓ Build tools installed${NC}"
echo ""

# ============================================================================
# SRT (Secure Reliable Transport) Library
# ============================================================================
echo -e "${YELLOW}[4/6] Installing SRT library...${NC}"
apt-get update
apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

apt-get install -y libsrt-openssl-dev libsrt-dev || {
    echo -e "${YELLOW}Warning: libsrt-dev not available in repositories${NC}"
    echo -e "${YELLOW}Attempting to build SRT from source...${NC}"
    
    # Install SRT build dependencies
    apt-get install -y \
        tclsh \
        libssl-dev \
        libtool \
        automake || exit 1
    
    # Build SRT from source
    cd /tmp
    git clone https://github.com/Haivision/srt.git
    cd srt
    
    ./configure
    make -j$(nproc)
    make install
    ldconfig
    
    cd /workspace
    rm -rf /tmp/srt
    
    echo -e "${GREEN}✓ SRT library built and installed from source${NC}"
}

echo -e "${GREEN}✓ SRT library installed${NC}"
echo ""

# ============================================================================
# Python Development Headers
# ============================================================================
echo -e "${YELLOW}[5/6] Installing Python development headers...${NC}"

apt-get install -y python3-dev || {
    echo -e "${RED}Failed to install Python development headers${NC}"
    exit 1
}

echo -e "${GREEN}✓ Python development headers installed${NC}"
echo ""

# ============================================================================
# Additional Utilities
# ============================================================================
echo -e "${YELLOW}[6/6] Installing additional utilities...${NC}"

UTILITIES=(
    wget                # Download tool
    curl                # HTTP client
    ca-certificates     # SSL certificates
    libboost-all-dev    # Boost C++ libraries (optional, for advanced features)
)

apt-get install -y "${UTILITIES[@]}" || {
    echo -e "${YELLOW}Warning: Some utilities failed to install (non-critical)${NC}"
}

echo -e "${GREEN}✓ Additional utilities installed${NC}"
echo ""

# ============================================================================
# Cleanup
# ============================================================================
echo -e "${YELLOW}Cleaning up...${NC}"

apt-get clean
rm -rf /var/lib/apt/lists/*

echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# ============================================================================
# Verify Installation
# ============================================================================
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Verifying Installation                               ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check critical libraries
CRITICAL_CHECKS=(
    "libGL.so.1:OpenGL"
    "libglib-2.0.so.0:GLib"
    "libsrt.so:SRT"
)

ALL_GOOD=true

for check in "${CRITICAL_CHECKS[@]}"; do
    IFS=':' read -r lib name <<< "$check"
    
    if ldconfig -p | grep -q "$lib"; then
        echo -e "${GREEN}✓${NC} $name library found"
    else
        echo -e "${RED}✗${NC} $name library NOT found"
        ALL_GOOD=false
    fi
done

echo ""

# Check build tools
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1)
    echo -e "${GREEN}✓${NC} GCC: $GCC_VERSION"
else
    echo -e "${RED}✗${NC} GCC not found"
    ALL_GOOD=false
fi

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo -e "${GREEN}✓${NC} $CMAKE_VERSION"
else
    echo -e "${RED}✗${NC} CMake not found"
    ALL_GOOD=false
fi

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found"
    ALL_GOOD=false
fi

echo ""

# Check Python packages (from Dockerfile)
echo "Checking Python packages from Dockerfile..."

PYTHON_PACKAGES=(
    "numpy"
    "opencv-python:cv2"
    "insightface"
    "onnxruntime-gpu:onnxruntime"
)

for pkg in "${PYTHON_PACKAGES[@]}"; do
    IFS=':' read -r pip_name import_name <<< "$pkg"
    import_name=${import_name:-$pip_name}  # Use pip_name if import_name not specified
    
    if python3 -c "import $import_name" 2>/dev/null; then
        VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'installed'))" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓${NC} $pip_name ($VERSION)"
    else
        echo -e "${RED}✗${NC} $pip_name not found"
        ALL_GOOD=false
    fi
done

echo ""

# ============================================================================
# Final Status
# ============================================================================
if [ "$ALL_GOOD" = true ]; then
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║  ✓ ALL REQUIREMENTS INSTALLED SUCCESSFULLY            ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}You can now build the SRT AI Server:${NC}"
    echo "  cd srt_ai_server"
    echo "  ./build.sh"
    echo ""
    exit 0
else
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║  ⚠ SOME REQUIREMENTS MISSING                          ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${YELLOW}Some dependencies are missing. Please check the errors above.${NC}"
    echo ""
    exit 1
fi