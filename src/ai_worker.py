"""
Ultra-Optimized Face Detection Module

This module is called from C++ with zero-copy NumPy arrays.
CRITICAL: Never modify input frames - they're shared memory from C++!
"""

import sys
import numpy as np
from insightface.app import FaceAnalysis

_app = None
_det_size = 640

def initialize(det_size=640):
    """
    Initialize InspireFace with TensorRT optimizations.
    
    Args:
        det_size: Detection input size (640 recommended)
    
    Returns:
        None (raises exception on failure)
    """
    global _app, _det_size
    
    print("[Python] Initializing InspireFace...", file=sys.stderr)
    
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_builder_optimization_level': 3,
        }),
        'CUDAExecutionProvider'
    ]
    
    try:
        _app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection']
        )
        
        _det_size = det_size
        _app.prepare(ctx_id=0, det_size=(det_size, det_size))
        
        print(f"[Python] ✓ Model loaded (det_size={det_size})", file=sys.stderr)
        
        print("[Python] Warming up TensorRT engine (20 iterations)...", file=sys.stderr)
        
        dummy_frame = np.zeros((det_size, det_size, 3), dtype=np.uint8)
        
        for i in range(20):
            _ = _app.get(dummy_frame)
            if i == 0:
                print("[Python]   - First run (engine building): may take 10-30s", file=sys.stderr)
            elif i % 5 == 0:
                print(f"[Python]   - Warmup {i}/20...", file=sys.stderr)
        
        print("[Python] ✓ Warmup complete! Server ready.", file=sys.stderr)
        
    except Exception as e:
        print(f"[Python] ERROR: Failed to initialize: {e}", file=sys.stderr)
        raise


def detect_faces(frame_np):
    """
    Perform face detection on a single frame.
    
    CRITICAL: frame_np is shared memory from C++ - DO NOT modify it!
    
    Args:
        frame_np: NumPy array (H, W, 3), dtype=uint8, BGR format
    
    Returns:
        List of dicts with bbox, kps, conf
    """
    if _app is None:
        raise RuntimeError("Model not initialized. Call initialize() first.")
    
    faces = _app.get(frame_np)
    
    results = []
    
    for face in faces:
        bbox = face.bbox.astype(np.int32).tolist()
        kps = face.kps.astype(np.int32).tolist() if hasattr(face, 'kps') and face.kps is not None else []
        conf = float(face.det_score) if hasattr(face, 'det_score') else 1.0
        
        results.append({
            'bbox': bbox,
            'kps': kps,
            'conf': conf
        })
    
    return results


if __name__ == "__main__":
    print("Testing ai_worker module...", file=sys.stderr)
    initialize(det_size=640)
    
    import cv2
    import time
    
    try:
        test_img = cv2.imread("test.jpg")
        if test_img is None:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            test_img = cv2.resize(test_img, (640, 640))
    except:
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
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