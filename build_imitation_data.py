#!/usr/bin/env python3
"""
모방학습 데이터 생성 파이프라인

비디오 → 포즈 추출(MediaPipe Tasks API) → State-Action 인코딩 → .npz 저장

생성되는 데이터 구조:
  - states:       [T, state_dim]   정규화된 관절 위치 + 속도
  - actions:      [T-1, action_dim] 프레임 간 위치 변화(delta)
  - poses:        [T, 33, 3]       정규화된 관절 좌표
  - velocity:     [T, 33, 3]       관절 속도
  - left_hand:    [T, 21, 3]       왼손 랜드마크
  - right_hand:   [T, 21, 3]       오른손 랜드마크
  - timestamps:   [T]              타임스탬프
  - confidence:   [T]              포즈 신뢰도
  - gripper_state:[T]              그리퍼(손 오므림) 상태 추정
"""

import os
import sys
import argparse
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# MediaPipe 포즈 추출 (Tasks API - 0.10.x)
# ============================================================================

def extract_pose_from_video(video_path: str, output_fps: float = 5.0, max_frames: int = None):
    """
    MediaPipe Tasks API로 비디오에서 포즈+손 추출
    
    Returns:
        dict with body[T,33,3], body_world[T,33,3],
             left_hand[T,21,3], right_hand[T,21,3],
             timestamps[T], confidence[T], fps
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as mp_base

    model_path = str(PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker.task")
    hand_model_path = str(PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task")

    # --- Pose Landmarker ---
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # --- Hand Landmarker ---
    hand_landmarker = None
    if Path(hand_model_path).exists():
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(orig_fps / output_fps))

    body_list = []
    body_world_list = []
    left_hand_list = []
    right_hand_list = []
    timestamps = []
    confidences = []

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if max_frames and processed >= max_frames:
            break

        timestamp_ms = int((frame_idx / orig_fps) * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 포즈 검출
        try:
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            frame_idx += 1
            continue

        # 포즈 데이터 추출
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            lms = pose_result.pose_landmarks[0]
            body = np.array([[l.x, l.y, l.z] for l in lms])
            conf = np.mean([l.visibility for l in lms]) if hasattr(lms[0], 'visibility') else 0.5
        else:
            body = np.zeros((33, 3))
            conf = 0.0

        # 월드 좌표
        if pose_result.pose_world_landmarks and len(pose_result.pose_world_landmarks) > 0:
            wlms = pose_result.pose_world_landmarks[0]
            body_w = np.array([[l.x, l.y, l.z] for l in wlms])
        else:
            body_w = np.zeros((33, 3))

        # 손 검출
        lh = np.zeros((21, 3))
        rh = np.zeros((21, 3))
        if hand_landmarker:
            try:
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                if hand_result.hand_landmarks:
                    for i, hand_lms in enumerate(hand_result.hand_landmarks):
                        hand_arr = np.array([[l.x, l.y, l.z] for l in hand_lms])
                        # handedness 확인
                        if hand_result.handedness and i < len(hand_result.handedness):
                            label = hand_result.handedness[i][0].category_name.lower()
                            if label == 'left':
                                lh = hand_arr
                            else:
                                rh = hand_arr
                        else:
                            if i == 0:
                                rh = hand_arr
                            else:
                                lh = hand_arr
            except Exception:
                pass

        body_list.append(body)
        body_world_list.append(body_w)
        left_hand_list.append(lh)
        right_hand_list.append(rh)
        timestamps.append(frame_idx / orig_fps)
        confidences.append(conf)

        frame_idx += 1
        processed += 1

    cap.release()
    pose_landmarker.close()
    if hand_landmarker:
        hand_landmarker.close()

    if not body_list:
        return None

    return {
        "body": np.array(body_list, dtype=np.float32),           # [T, 33, 3]
        "body_world": np.array(body_world_list, dtype=np.float32), # [T, 33, 3]
        "left_hand": np.array(left_hand_list, dtype=np.float32),   # [T, 21, 3]
        "right_hand": np.array(right_hand_list, dtype=np.float32), # [T, 21, 3]
        "timestamps": np.array(timestamps, dtype=np.float32),      # [T]
        "confidence": np.array(confidences, dtype=np.float32),     # [T]
        "fps": output_fps,
    }


# ============================================================================
# 모방학습 데이터 인코딩
# ============================================================================

def normalize_poses(poses: np.ndarray) -> np.ndarray:
    """포즈 정규화: hip 중심, 어깨 너비 scale"""
    # Hip center (23=left_hip, 24=right_hip)
    hip_center = (poses[:, 23, :] + poses[:, 24, :]) / 2
    normalized = poses - hip_center[:, np.newaxis, :]
    
    # 어깨 너비 기준 스케일링
    shoulder_width = np.linalg.norm(poses[:, 11, :] - poses[:, 12, :], axis=1)
    scale = np.clip(shoulder_width, 0.01, 2.0)
    normalized = normalized / scale[:, np.newaxis, np.newaxis]
    
    return normalized


def compute_velocity(poses: np.ndarray, fps: float) -> np.ndarray:
    """중앙 차분 속도 계산"""
    dt = 1.0 / fps
    vel = np.zeros_like(poses)
    if len(poses) > 2:
        vel[1:-1] = (poses[2:] - poses[:-2]) / (2 * dt)
        vel[0] = (poses[1] - poses[0]) / dt
        vel[-1] = (poses[-1] - poses[-2]) / dt
    elif len(poses) == 2:
        v = (poses[1] - poses[0]) / dt
        vel[0] = v
        vel[1] = v
    return vel


def estimate_gripper_state(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    손 오므림 정도로 그리퍼 상태 추정
    0.0 = 완전 열림, 1.0 = 완전 닫힘(쥐기)
    오른손 기준 (로봇팔 end-effector)
    """
    T = len(right_hand)
    gripper = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        hand = right_hand[t]
        if np.all(hand == 0):
            # 손 미검출 → 왼손 시도
            hand = left_hand[t]
            if np.all(hand == 0):
                gripper[t] = 0.5  # 불확실
                continue
        
        # 손가락 끝(4,8,12,16,20)과 손바닥(0) 사이 거리
        palm = hand[0]
        fingertips = hand[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(fingertips - palm, axis=1)
        avg_dist = np.mean(distances)
        
        # 정규화 (거리가 작을수록 닫힘)
        # 일반적으로 열린 손: 0.15~0.25, 닫힌 손: 0.03~0.08
        gripper[t] = np.clip(1.0 - (avg_dist - 0.03) / 0.20, 0.0, 1.0)
    
    return gripper


def build_states(norm_poses: np.ndarray, velocity: np.ndarray, 
                 confidence: np.ndarray) -> np.ndarray:
    """
    State 벡터 생성: [관절위치 flat | 관절속도 flat | 신뢰도]
    """
    T = norm_poses.shape[0]
    pos_flat = norm_poses.reshape(T, -1)    # [T, 99]
    vel_flat = velocity.reshape(T, -1)      # [T, 99]
    conf = confidence.reshape(T, 1)          # [T, 1]
    
    states = np.concatenate([pos_flat, vel_flat, conf], axis=1)  # [T, 199]
    return states.astype(np.float32)


def build_actions(norm_poses: np.ndarray, fps: float, 
                  gripper: np.ndarray) -> np.ndarray:
    """
    Action 벡터 생성: [관절위치 delta flat | gripper_state]
    delta = (pose[t+1] - pose[t]) * fps
    """
    T = norm_poses.shape[0]
    
    # 위치 변화량
    delta = np.diff(norm_poses, axis=0) * fps  # [T-1, 33, 3]
    delta_flat = delta.reshape(T - 1, -1)       # [T-1, 99]
    
    # 그리퍼 상태 (t+1 기준)
    grip = gripper[1:].reshape(T - 1, 1)
    
    actions = np.concatenate([delta_flat, grip], axis=1)  # [T-1, 100]
    return actions.astype(np.float32)


def encode_imitation_data(pose_data: dict, video_id: str) -> dict:
    """포즈 데이터 → 모방학습 데이터 인코딩"""
    body = pose_data["body"]           # [T, 33, 3]
    body_world = pose_data["body_world"]
    left_hand = pose_data["left_hand"]
    right_hand = pose_data["right_hand"]
    timestamps = pose_data["timestamps"]
    confidence = pose_data["confidence"]
    fps = pose_data["fps"]
    T = body.shape[0]
    
    if T < 3:
        raise ValueError(f"프레임 수 부족: {T}")
    
    # 1) 포즈 정규화
    norm_poses = normalize_poses(body)
    
    # 2) 속도 계산
    velocity = compute_velocity(norm_poses, fps)
    
    # 3) 그리퍼 상태 추정
    gripper = estimate_gripper_state(left_hand, right_hand)
    
    # 4) State 벡터 생성
    states = build_states(norm_poses, velocity, confidence)
    
    # 5) Action 벡터 생성
    actions = build_actions(norm_poses, fps, gripper)
    
    return {
        # 핵심 모방학습 데이터
        "states": states,                          # [T, 199]
        "actions": actions,                        # [T-1, 100]
        
        # 원시 포즈 데이터
        "poses": norm_poses.astype(np.float32),    # [T, 33, 3]
        "poses_raw": body.astype(np.float32),      # [T, 33, 3]
        "poses_world": body_world.astype(np.float32),
        "velocity": velocity.astype(np.float32),   # [T, 33, 3]
        
        # 손 데이터
        "left_hand": left_hand.astype(np.float32), # [T, 21, 3]
        "right_hand": right_hand.astype(np.float32),
        
        # 그리퍼 & 메타
        "gripper_state": gripper,                   # [T]
        "timestamps": timestamps,                   # [T]
        "confidence": confidence,                   # [T]
        
        # 메타데이터
        "fps": np.float32(fps),
        "video_id": str(video_id),
        "num_frames": np.int32(T),
        "state_dim": np.int32(states.shape[1]),
        "action_dim": np.int32(actions.shape[1]),
        "created_at": datetime.now().isoformat(),
    }


# ============================================================================
# 단일 비디오 처리 (프로세스 풀용)
# ============================================================================

def process_single_video(args_tuple):
    """단일 비디오: 포즈 추출 → 인코딩 → 저장"""
    video_path, output_dir, output_fps, max_frames, idx, total = args_tuple
    video_id = Path(video_path).stem
    output_path = Path(output_dir) / f"{video_id}_episode.npz"
    
    # 이미 모방학습 데이터가 있으면 스킵
    if output_path.exists():
        try:
            d = np.load(output_path, allow_pickle=True)
            if "states" in d and "actions" in d:
                return {"video_id": video_id, "status": "skipped", "msg": "already has IL data"}
        except:
            pass
    
    start = time.time()
    try:
        # 1) 포즈 추출
        pose_data = extract_pose_from_video(str(video_path), output_fps, max_frames)
        if pose_data is None:
            return {"video_id": video_id, "status": "failed", "msg": "no pose detected"}
        
        T = pose_data["body"].shape[0]
        if T < 3:
            return {"video_id": video_id, "status": "failed", "msg": f"too few frames: {T}"}
        
        # 2) 모방학습 데이터 인코딩
        il_data = encode_imitation_data(pose_data, video_id)
        
        # 3) 저장
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **il_data)
        
        elapsed = time.time() - start
        return {
            "video_id": video_id,
            "status": "success",
            "frames": int(T),
            "state_dim": int(il_data["state_dim"]),
            "action_dim": int(il_data["action_dim"]),
            "time": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "video_id": video_id,
            "status": "failed",
            "msg": str(e),
            "time": round(elapsed, 1),
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="모방학습 데이터 생성 파이프라인")
    parser.add_argument("--input-dir", default="data/raw", help="비디오 입력 디렉토리")
    parser.add_argument("--output-dir", default="data/episodes", help="에피소드 출력 디렉토리")
    parser.add_argument("--fps", type=float, default=5.0, help="추출 FPS (기본 5)")
    parser.add_argument("--max-frames", type=int, default=None, help="비디오당 최대 프레임")
    parser.add_argument("--limit", type=int, default=None, help="처리할 비디오 수 제한")
    parser.add_argument("--workers", type=int, default=1, help="병렬 워커 수")
    args = parser.parse_args()
    
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print("❌ 비디오 없음")
        return 1
    
    if args.limit:
        videos = videos[:args.limit]
    
    print(f"\n{'='*60}")
    print(f"🤖 모방학습 데이터 생성 파이프라인")
    print(f"{'='*60}")
    print(f"📹 비디오: {len(videos)}개")
    print(f"📁 출력:   {output_dir}")
    print(f"🎯 FPS:    {args.fps}")
    print(f"⚙️  워커:   {args.workers}")
    print(f"{'='*60}\n")
    
    tasks = [
        (str(v), str(output_dir), args.fps, args.max_frames, i, len(videos))
        for i, v in enumerate(videos)
    ]
    
    results = []
    success = 0
    failed = 0
    skipped = 0
    
    start_all = time.time()
    
    # 순차 처리 (MediaPipe는 프로세스별 모델 로딩 필요)
    for i, task in enumerate(tasks, 1):
        vid = Path(task[0]).stem
        print(f"[{i}/{len(tasks)}] {vid}...", end=" ", flush=True)
        
        result = process_single_video(task)
        results.append(result)
        
        if result["status"] == "success":
            success += 1
            print(f"✅ {result['frames']}f S:{result['state_dim']} A:{result['action_dim']} ({result['time']}s)")
        elif result["status"] == "skipped":
            skipped += 1
            print(f"⏭️  {result['msg']}")
        else:
            failed += 1
            print(f"❌ {result.get('msg','unknown')}")
    
    elapsed = time.time() - start_all
    
    print(f"\n{'='*60}")
    print(f"📊 결과 요약")
    print(f"{'='*60}")
    print(f"✅ 성공: {success}")
    print(f"⏭️  스킵: {skipped}")
    print(f"❌ 실패: {failed}")
    print(f"⏱️  소요: {elapsed:.1f}s")
    print(f"{'='*60}")
    
    # 결과 검증
    if success > 0:
        print(f"\n🔍 데이터 검증...")
        sample = list(Path(output_dir).glob("*_episode.npz"))
        if sample:
            d = np.load(sample[0], allow_pickle=True)
            print(f"  파일: {sample[0].name}")
            print(f"  키:   {list(d.keys())}")
            for k in ["states", "actions", "poses", "velocity", "gripper_state"]:
                if k in d:
                    print(f"  ✅ {k}: shape={d[k].shape}")
                else:
                    print(f"  ❌ {k}: 없음")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
