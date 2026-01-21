# src/ai_worker.py
"""
Ultra-Optimized Face Detection Module

This module is designed to be called from C++ with minimal overhead.
- Zero-copy NumPy arrays (shared memory with C++)
- Pre-loaded TensorRT model (no reload per frame)
- Minimal return data (only bbox + landmarks)

CRITICAL: Never modify input frames - they're shared memory from C++!
"""

import sys
import numpy as np
from insightface.app import FaceAnalysis

# Global model instance (loaded once at startup)
_app = None
_det_size = 640

def initialize(det_size=640):
    """
    Initialize InspireFace with TensorRT optimizations.
    
    This is called ONCE at server startup and performs:
    1. Model loading
    2. TensorRT engine building (if not cached)
    3. GPU warmup (20 iterations)
    
    Args:
        det_size: Detection input size (640 recommended for balance)
    
    Returns:
        None (raises exception on failure)
    """
    global _app, _det_size
    
    print("[Python] Initializing InspireFace...", file=sys.stderr)
    
    # TensorRT provider configuration
    # These settings are CRITICAL for performance
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,                          # GPU 0
            'trt_max_workspace_size': 2147483648,    # 2GB workspace
            'trt_fp16_enable': True,                 # FP16 = 2x faster
            'trt_engine_cache_enable': True,         # Cache engines
            'trt_engine_cache_path': './trt_cache',  # Cache directory
            'trt_builder_optimization_level': 3,     # Max optimization
        }),
        'CUDAExecutionProvider'  # Fallback to CUDA if TensorRT fails
    ]
    
    try:
        # Load ONLY detection module (no recognition, gender, age, etc.)
        # This keeps it lightweight and fast
        _app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection']
        )
        
        _det_size = det_size
        _app.prepare(ctx_id=0, det_size=(det_size, det_size))
        
        print(f"[Python] ✓ Model loaded (det_size={det_size})", file=sys.stderr)
        
        # Warmup: This is CRITICAL for consistent performance
        # First inference is slow (TensorRT engine building)
        # Warmup ensures GPU is at max clock speed
        print("[Python] Warming up TensorRT engine (20 iterations)...", file=sys.stderr)
        
        dummy_frame = np.zeros((det_size, det_size, 3), dtype=np.uint8)
        
        for i in range(20):
            _ = _app.get(dummy_frame)
            if i == 0:
                print("[Python]   - First run (engine building): may take 10-30s", file=sys.stderr)
            elif i % 5 == 0:
                print(f"[Python]   - Warmup {i}/20...", file=sys.stderr)
        
        print("[Python] ✓ Warmup complete! Server ready to accept connections.", file=sys.stderr)
        
    except Exception as e:
        print(f"[Python] ERROR: Failed to initialize: {e}", file=sys.stderr)
        raise


def detect_faces(frame_np):
    """
    Perform face detection on a single frame.
    
    CRITICAL: frame_np is shared memory from C++!
    - DO NOT modify it!
    - DO NOT store references to it!
    - Only read and return results
    
    Args:
        frame_np: NumPy array (H, W, 3), dtype=uint8, BGR format
                  This is SHARED MEMORY - no copy was made!
    
    Returns:
        List of dicts: [
            {
                'bbox': [x1, y1, x2, y2],  # Bounding box
                'kps': [[x,y], ...],       # 5 landmarks (eyes, nose, mouth corners)
                'conf': 0.98               # Confidence score
            },
            ...
        ]
    
    Performance: ~3.5ms on RTX 4080 (with TensorRT FP16)
    """
    if _app is None:
        raise RuntimeError("Model not initialized. Call initialize() first.")
    
    # Detect faces
    # InspireFace expects BGR format (which cv::Mat provides)
    faces = _app.get(frame_np)
    
    # Convert to minimal data structure for C++
    results = []
    
    for face in faces:
        # Extract bounding box (convert to integers for C++)
        bbox = face.bbox.astype(np.int32).tolist()
        
        # Extract keypoints/landmarks (5 points)
        kps = face.kps.astype(np.int32).tolist() if hasattr(face, 'kps') and face.kps is not None else []
        
        # Get confidence score (if available)
        conf = float(face.det_score) if hasattr(face, 'det_score') else 1.0
        
        results.append({
            'bbox': bbox,
            'kps': kps,
            'conf': conf
        })
    
    return results


# For testing Python module standalone
if __name__ == "__main__":
    print("Testing ai_worker module...", file=sys.stderr)
    
    # Initialize
    initialize(det_size=640)
    
    # Test with dummy frame
    import cv2
    
    # Try to load a test image
    try:
        test_img = cv2.imread("../image.jpg")
        if test_img is None:
            print("No test image found, using blank frame", file=sys.stderr)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            test_img = cv2.resize(test_img, (640, 640))
    except:
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Test detection
    import time
    
    print("\nRunning 10 test detections...", file=sys.stderr)
    times = []
    
    for i in range(10):
        start = time.perf_counter()
        results = detect_faces(test_img)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        print(f"Run {i+1}: {elapsed:.2f}ms, {len(results)} face(s)", file=sys.stderr)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    
    print(f"\n=== Results ===", file=sys.stderr)
    print(f"Average: {avg_time:.2f}ms", file=sys.stderr)
    print(f"Best: {min_time:.2f}ms", file=sys.stderr)
    print(f"FPS: {1000/avg_time:.1f}", file=sys.stderr)