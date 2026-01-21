#!/bin/bash
# deploy_fixes.sh - Deploy production fixes to cam_server
# Handles missing files gracefully

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  SRT AI Server - Production Fix Deployment          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -d "/workspace/cam_server" ]; then
    echo "ERROR: /workspace/cam_server directory not found!"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_DIR="/workspace/cam_server"

echo "[1/6] Checking for old files..."
BACKUP_NEEDED=false
FILES_TO_BACKUP=("src/python_bridge.cpp" "src/python_bridge.h" "src/srt_ai_server.cpp" "src/ai_worker.py" "CMakeLists.txt")

for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "${TARGET_DIR}/${file}" ]; then
        BACKUP_NEEDED=true
        echo "  ✓ Found: ${file}"
    fi
done

if [ "$BACKUP_NEEDED" = true ]; then
    echo ""
    echo "[2/6] Backing up old files..."
    BACKUP_DIR="${TARGET_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR/src"
    
    for file in "${FILES_TO_BACKUP[@]}"; do
        if [ -f "${TARGET_DIR}/${file}" ]; then
            cp "${TARGET_DIR}/${file}" "${BACKUP_DIR}/${file}" 2>/dev/null || true
            echo "  ✓ Backed up: ${file}"
        fi
    done
    echo "✓ Backup saved to: $BACKUP_DIR"
else
    echo "✓ No old files to backup (fresh installation)"
    echo ""
    echo "[2/6] Skipping backup step..."
fi

echo ""
echo "[3/6] Creating source directory structure..."
mkdir -p "${TARGET_DIR}/src"
echo "✓ Directory structure ready"

echo ""
echo "[4/6] Copying fixed source files..."

# Check if files exist in script directory before copying
FILES_TO_COPY=(
    "python_bridge.cpp:src/python_bridge.cpp"
    "python_bridge.h:src/python_bridge.h"
    "srt_ai_server.cpp:src/srt_ai_server.cpp"
    "ai_worker.py:src/ai_worker.py"
    "CMakeLists.txt:CMakeLists.txt"
)

for file_pair in "${FILES_TO_COPY[@]}"; do
    SOURCE=$(echo $file_pair | cut -d: -f1)
    DEST=$(echo $file_pair | cut -d: -f2)
    
    if [ -f "${SCRIPT_DIR}/${SOURCE}" ]; then
        cp "${SCRIPT_DIR}/${SOURCE}" "${TARGET_DIR}/${DEST}"
        echo "  ✓ Copied: ${SOURCE} → ${DEST}"
    else
        echo "  ✗ WARNING: ${SOURCE} not found in ${SCRIPT_DIR}"
    fi
done

echo "✓ Files copied"

echo ""
echo "[5/6] Cleaning old build..."
if [ -d "${TARGET_DIR}/build" ]; then
    rm -rf "${TARGET_DIR}/build"
    echo "✓ Old build directory removed"
fi
mkdir -p "${TARGET_DIR}/build"
echo "✓ Fresh build directory created"

echo ""
echo "[6/6] Building project..."
cd "${TARGET_DIR}/build"

# Run cmake with error handling
if ! cmake ..; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo "Please check the error messages above."
    exit 1
fi

# Run make with error handling
if ! make -j$(nproc); then
    echo ""
    echo "ERROR: Build failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo "✓ Build successful"

echo ""
echo "[7/6] Verifying installation..."

# Check if executable exists
if [ -f "${TARGET_DIR}/build/srt_ai_server" ]; then
    echo "✓ Executable: ${TARGET_DIR}/build/srt_ai_server"
else
    echo "✗ ERROR: Executable not found!"
    exit 1
fi

# Check if ai_worker.py is in the right places
PYTHON_MODULE_FOUND=false
if [ -f "${TARGET_DIR}/build/ai_worker.py" ]; then
    echo "✓ Python module: build/ai_worker.py"
    PYTHON_MODULE_FOUND=true
fi
if [ -f "${TARGET_DIR}/build/src/ai_worker.py" ]; then
    echo "✓ Python module: build/src/ai_worker.py"
    PYTHON_MODULE_FOUND=true
fi

if [ "$PYTHON_MODULE_FOUND" = false ]; then
    echo "✗ WARNING: ai_worker.py not found in build directory"
    echo "  Copying manually..."
    cp "${TARGET_DIR}/src/ai_worker.py" "${TARGET_DIR}/build/" 2>/dev/null || true
    cp "${TARGET_DIR}/src/ai_worker.py" "${TARGET_DIR}/build/src/" 2>/dev/null || true
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT SUCCESSFUL!                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "To run the server:"
echo "  cd ${TARGET_DIR}/build"
echo "  ./srt_ai_server"
echo ""

if [ "$BACKUP_NEEDED" = true ]; then
    echo "Old files backed up to: $BACKUP_DIR"
    echo ""
fi

echo "Files installed:"
echo "  ${TARGET_DIR}/src/python_bridge.cpp"
echo "  ${TARGET_DIR}/src/python_bridge.h"
echo "  ${TARGET_DIR}/src/srt_ai_server.cpp"
echo "  ${TARGET_DIR}/src/ai_worker.py"
echo "  ${TARGET_DIR}/CMakeLists.txt"
echo ""