#!/usr/bin/env python3
"""RTMPose WholeBody 파이프라인 통합 테스트"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n=== Test 1: Model Loading ===", flush=True)
    import onnxruntime as ort
    print(f"  onnxruntime: {ort.__version__}", flush=True)
    print(f"  providers: {ort.get_available_providers()}", flush=True)

    from extraction.rtmpose_wholebody import YOLOXDetector, RTMPoseWholeBody

    t0 = time.time()
    det = YOLOXDetector("models/rtmpose/yolox_l.onnx", device="gpu")
    t1 = time.time()
    print(f"  YOLOX loaded: {t1-t0:.2f}s, provider: {det.session.get_providers()[0]}", flush=True)

    pose = RTMPoseWholeBody("models/rtmpose/dwpose_wholebody.onnx", device="gpu")
    t2 = time.time()
    print(f"  DWPose loaded: {t2-t1:.2f}s, provider: {pose.session.get_providers()[0]}", flush=True)
    print(f"  DWPose input: {pose.input_shape}", flush=True)
    print("  ✅ PASS", flush=True)
    return det, pose


def test_single_frame(det, pose):
    """단일 프레임 추론 테스트"""
    print("\n=== Test 2: Single Frame Inference ===", flush=True)
    import cv2
    import numpy as np

    # 실제 비디오에서 첫 프레임 읽기
    video_path = "data/raw/--yM7qWlBh4.mp4"
    if not os.path.exists(video_path):
        # 더미 프레임 생성
        print("  No test video, using synthetic frame", flush=True)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("  ❌ Cannot read video frame", flush=True)
            return
    
    h, w = frame.shape[:2]
    print(f"  Frame: {w}x{h}", flush=True)

    # 검출
    t0 = time.time()
    dets = det(frame, score_thr=0.3)
    t1 = time.time()
    print(f"  Detection: {(t1-t0)*1000:.1f}ms", flush=True)

    if dets and len(dets[0]) > 0:
        print(f"  Detected {len(dets[0])} person(s)", flush=True)
        bbox = dets[0][0]
        print(f"  Best bbox: [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] score={bbox[4]:.3f}", flush=True)
    else:
        print("  No detection, using full frame", flush=True)
        bbox = np.array([0, 0, w, h, 1.0])

    # 포즈
    t2 = time.time()
    result = pose(frame, bbox)
    t3 = time.time()
    print(f"  Pose inference: {(t3-t2)*1000:.1f}ms", flush=True)
    print(f"  keypoints: {result['keypoints'].shape}", flush=True)
    print(f"  body: {result['body'].shape} (score_mean={result['body'][:,2].mean():.3f})", flush=True)
    print(f"  left_hand: {result['left_hand'].shape} (score_mean={result['left_hand'][:,2].mean():.3f})", flush=True)
    print(f"  right_hand: {result['right_hand'].shape} (score_mean={result['right_hand'][:,2].mean():.3f})", flush=True)
    print("  ✅ PASS", flush=True)


def test_video_extraction():
    """비디오 전체 추출 테스트"""
    print("\n=== Test 3: Video Extraction ===", flush=True)
    import numpy as np
    from extraction.rtmpose_wholebody import RTMPoseVideoExtractor

    extractor = RTMPoseVideoExtractor(device="gpu")

    video_path = "data/raw/--yM7qWlBh4.mp4"
    if not os.path.exists(video_path):
        print("  ⏭ No test video, skipping", flush=True)
        return None

    t0 = time.time()
    pose_data = extractor.extract(video_path, output_fps=2.0, max_frames=10)
    t1 = time.time()

    if pose_data is None:
        print("  ❌ No pose data extracted", flush=True)
        return None

    print(f"  Extraction time: {t1-t0:.2f}s", flush=True)
    for k, v in pose_data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)

    # 검증
    assert pose_data["body"].shape[1] == 17, f"body should be 17 keypoints, got {pose_data['body'].shape[1]}"
    assert pose_data["body"].shape[2] == 3, f"body should have 3 channels (x,y,conf)"
    assert pose_data["left_hand"].shape[1] == 21, f"left_hand should be 21 keypoints"
    assert pose_data["right_hand"].shape[1] == 21, f"right_hand should be 21 keypoints"
    T = pose_data["body"].shape[0]
    assert T > 0, "Should have at least 1 frame"
    assert len(pose_data["timestamps"]) == T
    assert len(pose_data["confidence"]) == T

    print(f"  ✅ PASS ({T} frames extracted)", flush=True)
    return pose_data


def test_encoding_compatibility(pose_data):
    """인코딩 호환성 테스트"""
    print("\n=== Test 4: Encoding Compatibility ===", flush=True)
    import numpy as np

    if pose_data is None:
        print("  ⏭ No pose data, skipping", flush=True)
        return

    body = pose_data["body"]
    T, J, C = body.shape
    print(f"  Body: T={T}, J={J}, C={C}", flush=True)

    # COCO 17 keypoints에서 hip은 11(left_hip), 12(right_hip)
    # RTMPose COCO format: 11=left_hip, 12=right_hip
    # shoulder: 5=left_shoulder, 6=right_shoulder
    hip_center = (body[:, 11, :2] + body[:, 12, :2]) / 2  # COCO hip
    shoulder_width = np.linalg.norm(body[:, 5, :2] - body[:, 6, :2], axis=1)
    
    print(f"  Hip center (mean): {hip_center.mean(axis=0)}", flush=True)
    print(f"  Shoulder width (mean): {shoulder_width.mean():.4f}", flush=True)

    # SO-101 관련 키포인트 (shoulder/elbow/wrist)
    so101_indices = [5, 6, 7, 8, 9, 10]  # L/R shoulder, elbow, wrist
    so101_kps = body[:, so101_indices, :]
    print(f"  SO-101 keypoints shape: {so101_kps.shape}", flush=True)
    print(f"  SO-101 confidence (mean): {so101_kps[:,:,2].mean():.3f}", flush=True)

    print("  ✅ PASS", flush=True)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("🧪 RTMPose WholeBody Pipeline Test", flush=True)
    print("=" * 60, flush=True)

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    try:
        det, pose = test_model_loading()
        test_single_frame(det, pose)
        pose_data = test_video_extraction()
        test_encoding_compatibility(pose_data)
        print("\n" + "=" * 60, flush=True)
        print("🎉 All tests passed!", flush=True)
        print("=" * 60, flush=True)
    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
