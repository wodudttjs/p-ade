#!/usr/bin/env python3
"""RTMPose → 인코딩 전체 파이프라인 통합 테스트"""
import sys, os, time
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

def test_full_pipeline():
    """비디오 → 포즈 추출 → 인코딩 → 검증"""
    print("=" * 60, flush=True)
    print("🧪 Full Pipeline Test: Video → Pose → Encode", flush=True)
    print("=" * 60, flush=True)

    video_path = "data/raw/--yM7qWlBh4.mp4"
    if not os.path.exists(video_path):
        print("⏭ No test video, skipping", flush=True)
        return

    # 1) 포즈 추출
    print("\n[Step 1] RTMPose WholeBody 포즈 추출...", flush=True)
    from extraction.rtmpose_wholebody import RTMPoseVideoExtractor
    extractor = RTMPoseVideoExtractor(device="gpu")
    
    t0 = time.time()
    pose_data = extractor.extract(video_path, output_fps=2.0, max_frames=20)
    t1 = time.time()
    
    assert pose_data is not None, "Pose extraction returned None"
    T = pose_data["body"].shape[0]
    print(f"  ✅ {T} frames, {t1-t0:.2f}s", flush=True)
    print(f"  body: {pose_data['body'].shape}", flush=True)
    print(f"  left_hand: {pose_data['left_hand'].shape}", flush=True)
    print(f"  right_hand: {pose_data['right_hand'].shape}", flush=True)

    # 2) 인코딩
    print("\n[Step 2] 모방학습 데이터 인코딩...", flush=True)
    sys.path.insert(0, "scripts/pipeline")
    from build_imitation_data import encode_imitation_data
    
    t2 = time.time()
    il_data = encode_imitation_data(pose_data, "test_video")
    t3 = time.time()
    print(f"  ✅ 인코딩 완료 {t3-t2:.3f}s", flush=True)

    # 3) 검증
    print("\n[Step 3] 데이터 검증...", flush=True)
    
    # 핵심 차원 검증
    states = il_data["states"]
    actions = il_data["actions"]
    poses = il_data["poses"]
    velocity = il_data["velocity"]
    gripper = il_data["gripper_state"]
    
    print(f"  states:       {states.shape} (expect [T, 103])", flush=True)
    print(f"  actions:      {actions.shape} (expect [T-1, 52])", flush=True)
    print(f"  poses:        {poses.shape} (expect [T, 17, 3])", flush=True)
    print(f"  velocity:     {velocity.shape} (expect [T, 17, 3])", flush=True)
    print(f"  gripper:      {gripper.shape} (expect [T])", flush=True)
    print(f"  left_hand:    {il_data['left_hand'].shape} (expect [T, 21, 3])", flush=True)
    print(f"  right_hand:   {il_data['right_hand'].shape} (expect [T, 21, 3])", flush=True)

    # 차원 정확성
    assert states.shape == (T, 103), f"states shape mismatch: {states.shape}"
    assert actions.shape == (T-1, 52), f"actions shape mismatch: {actions.shape}"
    assert poses.shape == (T, 17, 3), f"poses shape mismatch: {poses.shape}"
    assert velocity.shape == (T, 17, 3), f"velocity shape mismatch: {velocity.shape}"
    assert gripper.shape == (T,), f"gripper shape mismatch: {gripper.shape}"

    # 값 범위 검증
    print(f"\n  States range:  [{states.min():.3f}, {states.max():.3f}]", flush=True)
    print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]", flush=True)
    print(f"  Gripper range: [{gripper.min():.3f}, {gripper.max():.3f}]", flush=True)
    
    assert 0.0 <= gripper.min() and gripper.max() <= 1.0, "Gripper out of [0,1]"
    assert not np.any(np.isnan(states)), "NaN in states"
    assert not np.any(np.isnan(actions)), "NaN in actions"

    # 4) .npz 저장/로드 테스트
    print("\n[Step 4] NPZ 저장/로드 테스트...", flush=True)
    os.makedirs("/tmp/pade_test_episodes", exist_ok=True)
    npz_path = "/tmp/pade_test_episodes/test_episode.npz"
    np.savez_compressed(npz_path, **il_data)
    loaded = dict(np.load(npz_path, allow_pickle=True))
    
    for key in ["states", "actions", "poses", "velocity", "gripper_state"]:
        assert key in loaded, f"Missing key: {key}"
        orig = il_data[key] if isinstance(il_data[key], np.ndarray) else np.array(il_data[key])
        load = loaded[key]
        if orig.dtype.kind == 'f':
            assert np.allclose(orig, load, atol=1e-5), f"Data mismatch for {key}"
    
    print(f"  ✅ NPZ round-trip OK ({os.path.getsize(npz_path)/1024:.1f}KB)", flush=True)

    # 메타데이터 확인
    print(f"\n  state_dim: {il_data['state_dim']}", flush=True)
    print(f"  action_dim: {il_data['action_dim']}", flush=True)
    print(f"  fps: {il_data['fps']}", flush=True)
    print(f"  num_frames: {il_data['num_frames']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("🎉 Full pipeline test PASSED!", flush=True)
    print("=" * 60, flush=True)

    # 정리
    os.remove(npz_path)
    os.rmdir("/tmp/pade_test_episodes")


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        import traceback
        print(f"\n❌ Test FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
