#!/bin/bash
# deploy_fixes.sh - Deploy production fixes to cam_server

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

echo "[1/5] Backing up old files..."
BACKUP_DIR="${TARGET_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/src"
cp "${TARGET_DIR}/src/python_bridge.cpp" "$BACKUP_DIR/src/" 2>/dev/null || true
cp "${TARGET_DIR}/src/python_bridge.h" "$BACKUP_DIR/src/" 2>/dev/null || true
cp "${TARGET_DIR}/src/srt_ai_server.cpp" "$BACKUP_DIR/src/" 2>/dev/null || true
cp "${TARGET_DIR}/src/ai_worker.py" "$BACKUP_DIR/src/" 2>/dev/null || true
echo "✓ Backed up to: $BACKUP_DIR"

echo ""
echo "[2/5] Copying fixed source files..."
cp "${SCRIPT_DIR}/python_bridge.cpp" "${TARGET_DIR}/src/"
cp "${SCRIPT_DIR}/python_bridge.h" "${TARGET_DIR}/src/"
cp "${SCRIPT_DIR}/srt_ai_server.cpp" "${TARGET_DIR}/src/"
cp "${SCRIPT_DIR}/ai_worker.py" "${TARGET_DIR}/src/"
echo "✓ Files copied"

echo ""
echo "[3/5] Cleaning old build..."
rm -rf "${TARGET_DIR}/build"
mkdir -p "${TARGET_DIR}/build"
echo "✓ Build directory cleaned"

echo ""
echo "[4/5] Building project..."
cd "${TARGET_DIR}/build"
cmake .. && make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi
echo "✓ Build successful"

echo ""
echo "[5/5] Verifying executable..."
if [ -f "${TARGET_DIR}/build/srt_ai_server" ]; then
    echo "✓ Executable created: ${TARGET_DIR}/build/srt_ai_server"
else
    echo "ERROR: Executable not found!"
    exit 1
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
echo "Old files backed up to: $BACKUP_DIR"
echo ""