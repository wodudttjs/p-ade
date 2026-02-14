#!/usr/bin/env python
"""
Action Encoding CLI

MVP Phase 2 Week 7: Action Encoding
- State-Action 쌍 생성
- Delta position 계산
- 표준화된 데이터 포맷

사용법:
    python encode_actions.py --all                    # 모든 포즈 변환
    python encode_actions.py --file pose.npz          # 단일 파일 변환
    python encode_actions.py --all --output episodes  # 출력 디렉토리 지정
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import shutil

import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from transformation.encoding import StateBuilder, ActionComputer, StateActionPair
from transformation.spec import TransformConfig, StateSpec, ActionSpec

logger = setup_logger(__name__)


@dataclass
class EncodingResult:
    """인코딩 결과"""
    file_path: str
    video_id: str
    success: bool
    output_path: Optional[str] = None
    num_frames: int = 0
    state_dim: int = 0
    action_dim: int = 0
    error: Optional[str] = None


class ActionEncoder:
    """Action Encoding 클래스"""
    
    def __init__(
        self,
        poses_dir: str = "data/poses",
        output_dir: str = "data/episodes",
        config: Optional[TransformConfig] = None,
    ):
        self.poses_dir = Path(poses_dir)
        self.output_dir = Path(output_dir)
        self.config = config or TransformConfig()
        
        # State/Action builders
        self.state_builder = StateBuilder(
            config=self.config,
            state_spec=StateSpec(
                joint_positions=True,
                joint_velocities=True,
                object_relations=False,
                confidence_stats=True,
            ),
        )
        
        self.action_computer = ActionComputer(
            config=self.config,
            action_spec=ActionSpec(
                position_delta=True,
                rotation_delta=False,
                gripper_state=False,
                eef_only=True,
            ),
        )
    
    def load_pose_data(self, file_path: Path) -> Dict[str, np.ndarray]:
        """포즈 데이터 로드"""
        data = np.load(file_path, allow_pickle=True)
        
        result = {}
        
        # Body poses
        for key in ["poses", "body", "keypoints"]:
            if key in data:
                result["poses"] = data[key]
                break
        
        # Hand landmarks
        if "left_hand" in data and "right_hand" in data:
            left = data["left_hand"]
            right = data["right_hand"]
            if len(left.shape) == 3 and len(right.shape) == 3:
                result["hands"] = np.stack([left, right], axis=1)
        
        # Confidence
        for key in ["confidences", "confidence", "conf"]:
            if key in data:
                result["confidence"] = data[key]
                break
        
        # Timestamps
        if "timestamps" in data:
            result["timestamps"] = data["timestamps"]
        
        # FPS
        if "fps" in data:
            result["fps"] = float(data["fps"])
        elif "metadata" in data:
            meta = data["metadata"]
            if hasattr(meta, "__len__") and len(meta) >= 1:
                result["fps"] = float(meta[0]) if meta[0] > 0 else 30.0
            else:
                result["fps"] = 30.0
        else:
            result["fps"] = 30.0
        
        return result
    
    def compute_velocity(self, poses: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """속도 계산"""
        dt = 1.0 / fps
        
        # 중앙 차분 (central difference)
        velocity = np.zeros_like(poses)
        velocity[1:-1] = (poses[2:] - poses[:-2]) / (2 * dt)
        velocity[0] = (poses[1] - poses[0]) / dt
        velocity[-1] = (poses[-1] - poses[-2]) / dt
        
        return velocity
    
    def normalize_poses(self, poses: np.ndarray) -> np.ndarray:
        """포즈 정규화 (hip 중심, scale 정규화)"""
        # Hip center (MediaPipe: 23=left_hip, 24=right_hip)
        if poses.shape[1] >= 25:
            hip_center = (poses[:, 23, :] + poses[:, 24, :]) / 2
        else:
            hip_center = np.mean(poses, axis=1)
        
        # 중심 이동
        normalized = poses - hip_center[:, np.newaxis, :]
        
        # Scale 정규화 (어깨 너비 기준)
        if poses.shape[1] >= 12:
            shoulder_width = np.linalg.norm(
                poses[:, 11, :] - poses[:, 12, :], axis=1
            )
            scale = np.clip(shoulder_width, 0.1, 2.0)
            normalized = normalized / scale[:, np.newaxis, np.newaxis]
        
        return normalized
    
    def encode_file(self, file_path: Path) -> EncodingResult:
        """단일 파일 인코딩"""
        video_id = file_path.stem.replace("_pose", "")
        
        try:
            # 데이터 로드
            data = self.load_pose_data(file_path)
            
            if "poses" not in data:
                raise ValueError("포즈 데이터 없음")
            
            poses = data["poses"]
            fps = data.get("fps", 30.0)
            confidence = data.get("confidence", None)
            hands = data.get("hands", None)
            timestamps = data.get("timestamps", None)
            
            T = poses.shape[0]
            
            if T < 2:
                raise ValueError(f"프레임 수 부족: {T}")
            
            # 포즈 정규화
            normalized_poses = self.normalize_poses(poses)
            
            # 속도 계산
            velocity = self.compute_velocity(normalized_poses, fps)
            
            # State 생성
            states, state_masks = self.state_builder.build_state(
                pose=normalized_poses,
                velocity=velocity,
                conf=confidence,
            )
            
            # Action 계산
            actions, action_masks = self.action_computer.compute_action(
                pose=normalized_poses,
                dt=1.0 / fps,
            )
            
            # Gripper state 추정 (손 데이터가 있는 경우)
            gripper_states = None
            if hands is not None:
                gripper_states = self.action_computer.estimate_gripper_state(hands)
            
            # Timestamps 생성
            if timestamps is None:
                timestamps = np.arange(T) / fps
            
            # 출력 저장
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{video_id}_episode.npz"
            
            save_dict = {
                # State-Action pairs
                "states": states.astype(np.float32),
                "actions": actions.astype(np.float32),
                "state_masks": state_masks,
                "action_masks": action_masks,
                
                # Raw data
                "poses": normalized_poses.astype(np.float32),
                "velocity": velocity.astype(np.float32),
                "timestamps": timestamps.astype(np.float32),
                
                # Metadata
                "fps": fps,
                "video_id": video_id,
                "state_dim": states.shape[1],
                "action_dim": actions.shape[1],
                "num_frames": T,
            }
            
            if confidence is not None:
                save_dict["confidence"] = confidence.astype(np.float32)
            
            if gripper_states is not None:
                save_dict["gripper_states"] = gripper_states.astype(np.float32)
            
            np.savez_compressed(output_path, **save_dict)
            
            return EncodingResult(
                file_path=str(file_path),
                video_id=video_id,
                success=True,
                output_path=str(output_path),
                num_frames=T,
                state_dim=states.shape[1],
                action_dim=actions.shape[1],
            )
            
        except Exception as e:
            logger.error(f"인코딩 실패 {file_path}: {e}")
            return EncodingResult(
                file_path=str(file_path),
                video_id=video_id,
                success=False,
                error=str(e),
            )
    
    def encode_all(self) -> List[EncodingResult]:
        """모든 포즈 파일 인코딩"""
        results = []
        
        pose_files = list(self.poses_dir.glob("*_pose.npz"))
        
        if not pose_files:
            logger.warning(f"포즈 파일 없음: {self.poses_dir}")
            return results
        
        print(f"\n{'='*60}")
        print(f"🎬 Action Encoding 시작")
        print(f"{'='*60}")
        print(f"📁 입력: {self.poses_dir}")
        print(f"📁 출력: {self.output_dir}")
        print(f"📦 파일: {len(pose_files)}개")
        print()
        
        for i, file_path in enumerate(pose_files, 1):
            result = self.encode_file(file_path)
            results.append(result)
            
            if result.success:
                status = f"✅ ({result.num_frames} frames, S:{result.state_dim}, A:{result.action_dim})"
            else:
                status = f"❌ {result.error}"
            
            print(f"[{i}/{len(pose_files)}] {result.video_id}: {status}")
        
        return results
    
    def print_summary(self, results: List[EncodingResult]):
        """요약 출력"""
        if not results:
            print("결과 없음")
            return
        
        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print()
        print("="*60)
        print("📊 인코딩 결과 요약")
        print("="*60)
        
        print(f"\n📈 요약:")
        print(f"   총 파일: {len(results)}개")
        print(f"   ✅ 성공: {len(success)}개 ({len(success)/len(results)*100:.1f}%)")
        print(f"   ❌ 실패: {len(failed)}개")
        
        if success:
            total_frames = sum(r.num_frames for r in success)
            avg_state_dim = sum(r.state_dim for r in success) / len(success)
            avg_action_dim = sum(r.action_dim for r in success) / len(success)
            
            print(f"\n📊 통계:")
            print(f"   총 프레임: {total_frames}")
            print(f"   평균 State 차원: {avg_state_dim:.0f}")
            print(f"   평균 Action 차원: {avg_action_dim:.0f}")
        
        if failed:
            print(f"\n❌ 실패 목록:")
            for r in failed:
                print(f"   - {r.video_id}: {r.error}")
        
        print()


def inspect_episode(file_path: str):
    """에피소드 파일 상세 검사"""
    data = np.load(file_path)
    
    print(f"\n📄 파일: {file_path}")
    print(f"{'='*60}")
    
    print("\n📦 데이터 키:")
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"   {key}: {arr}")
    
    if "states" in data:
        states = data["states"]
        print(f"\n🎯 States:")
        print(f"   Shape: {states.shape}")
        print(f"   Mean: {np.mean(states):.4f}")
        print(f"   Std: {np.std(states):.4f}")
        print(f"   Min: {np.min(states):.4f}")
        print(f"   Max: {np.max(states):.4f}")
    
    if "actions" in data:
        actions = data["actions"]
        print(f"\n🚀 Actions:")
        print(f"   Shape: {actions.shape}")
        print(f"   Mean: {np.mean(actions):.6f}")
        print(f"   Std: {np.std(actions):.6f}")
        print(f"   Min: {np.min(actions):.6f}")
        print(f"   Max: {np.max(actions):.6f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="P-ADE Action Encoding")
    
    parser.add_argument("--all", action="store_true", help="모든 포즈 파일 인코딩")
    parser.add_argument("--file", help="단일 파일 인코딩")
    parser.add_argument("--inspect", help="에피소드 파일 검사")
    
    parser.add_argument("--poses-dir", default="data/poses", help="포즈 디렉토리")
    parser.add_argument("--output-dir", default="data/episodes", help="출력 디렉토리")
    
    parser.add_argument("--eef-only", action="store_true", default=True,
                        help="End-effector만 사용 (기본값)")
    parser.add_argument("--all-joints", action="store_true",
                        help="전체 관절 사용")
    parser.add_argument("--no-gpu-streams", action="store_true",
                        help="GPU 3-Stream 비활성화")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_episode(args.inspect)
        return
    
    # 인코더 생성
    encoder = ActionEncoder(
        poses_dir=args.poses_dir,
        output_dir=args.output_dir,
    )
    
    if args.all_joints:
        encoder.action_computer.action_spec.eef_only = False
    
    results = []
    
    if args.file:
        result = encoder.encode_file(Path(args.file))
        results = [result]
        
        if result.success:
            print(f"\n✅ 인코딩 완료: {result.output_path}")
            print(f"   프레임: {result.num_frames}")
            print(f"   State 차원: {result.state_dim}")
            print(f"   Action 차원: {result.action_dim}")
        else:
            print(f"\n❌ 인코딩 실패: {result.error}")
    
    elif args.all:
        # GPU 3-Stream 병렬 처리 시도
        gpu_used = False
        if not args.no_gpu_streams:
            try:
                from gpu.stream_manager import GPU3StreamManager
                stream_mgr = GPU3StreamManager()
                batch_size = stream_mgr.auto_adjust_batch_size()
                vram = stream_mgr.get_vram_usage()
                print(f"\n🎮 GPU 3-Stream 활성화 (배치: {batch_size}, VRAM: {vram.get('allocated', 0):.1f}GB)")
                
                pose_files = list(Path(args.poses_dir).glob("*_pose.npz"))
                if pose_files:
                    processor = stream_mgr.make_encode_processor(
                        poses_dir=args.poses_dir,
                        output_dir=args.output_dir,
                    )
                    gpu_results = stream_mgr.process_batch(
                        [str(f) for f in pose_files], processor
                    )
                    
                    for r in gpu_results:
                        if r and r.get("success"):
                            if r.get("status") == "skipped":
                                print(f"  ⏭️  {r.get('video_id', '?')}: {r.get('msg', 'skipped')}")
                            else:
                                print(f"  ✅ {r.get('video_id', '?')}: "
                                      f"{r.get('frames', 0)}f S:{r.get('state_dim', '?')} A:{r.get('action_dim', '?')}")
                            results.append(EncodingResult(
                                file_path=r.get("file_path", ""),
                                video_id=r.get("video_id", ""),
                                success=True,
                                num_frames=r.get("frames", 0),
                                state_dim=r.get("state_dim", 0),
                                action_dim=r.get("action_dim", 0),
                            ))
                        else:
                            print(f"  ❌ {r.get('video_id', '?')}: {r.get('error', 'unknown')}")
                            results.append(EncodingResult(
                                file_path=r.get("file_path", ""),
                                video_id=r.get("video_id", ""),
                                success=False,
                                error=r.get("error", "unknown"),
                            ))
                    
                    stream_mgr.print_stats()
                    gpu_used = True
                else:
                    print(f"⚠️ 포즈 파일 없음: {args.poses_dir}")
            except Exception as e:
                print(f"⚠️ GPU 3-Stream 실패, 순차 모드로 폴백: {e}")
        
        # 폴백: 순차 처리
        if not gpu_used:
            results = encoder.encode_all()
        encoder.print_summary(results)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
