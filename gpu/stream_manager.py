"""
GPU 3-Stream 병렬 처리 매니저

CUDA Stream을 활용한 3개 영상 동시 처리를 구현합니다.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)

# PyTorch/CUDA 임포트
try:
    import torch
    import torch.cuda as cuda
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    torch = None


@dataclass
class StreamConfig:
    """스트림 설정"""
    num_streams: int = 6              # 동시 스트림 수 (dual-GPU: 3+3)
    gpu_ids: List[int] = None         # 사용할 GPU ID 목록 (None=자동)
    streams_per_gpu: int = 3          # GPU당 스트림 수
    vram_limit_ratio: float = 0.85    # 총 VRAM의 85% 사용
    vram_limit_gb: float = 9.0        # 단일 GPU VRAM 한계 (레거시 호환)
    target_fps: int = 30              # 기본 타겟 FPS
    long_video_fps: int = 15          # 긴 영상(60초+) FPS
    long_video_threshold_sec: int = 600  # 10분 이하는 GPU, 초과만 CPU

    def __post_init__(self):
        if self.gpu_ids is None:
            if CUDA_AVAILABLE:
                n = torch.cuda.device_count()
                self.gpu_ids = list(range(min(n, 2)))  # 최대 2개 GPU
            else:
                self.gpu_ids = [0]


class GPU3StreamManager:
    """
    GPU 6-Stream 병렬 처리 매니저 (Dual-GPU 지원)

    CUDA Stream을 사용하여 최대 6개의 영상을 동시에 처리합니다.
    GPU가 1개면 3-Stream, 2개 이상이면 6-Stream(각 GPU에 3개).

    영상 길이별 라우팅:
      - < 30초:  batch 4 (짧은 영상)
      - 30-60초: batch 2 (중간 영상)
      - > 60초:  CPUWorkerPool에 위임

    사용법:
        manager = GPU3StreamManager()
        results = manager.process_batch(video_paths, processor)
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._lock = threading.Lock()

        # 통계
        self._stats = {
            "total_processed": 0,
            "total_time_sec": 0,
            "peak_vram_gb": 0,
        }

        # CUDA 스트림: {gpu_id: [Stream, ...]}
        self._streams: Dict[int, List] = {}
        self._devices: List = []

        if CUDA_AVAILABLE:
            self._init_cuda()
        else:
            logger.warning("⚠️ CUDA 사용 불가 - CPU 모드로 동작")

    def _init_cuda(self):
        """Dual-GPU CUDA 초기화"""
        available_gpus = torch.cuda.device_count()
        gpu_ids = [g for g in self.config.gpu_ids if g < available_gpus]

        if not gpu_ids:
            gpu_ids = [0]

        for gpu_id in gpu_ids:
            device = torch.device(f"cuda:{gpu_id}")
            self._devices.append(device)
            with torch.cuda.device(gpu_id):
                streams = [cuda.Stream() for _ in range(self.config.streams_per_gpu)]
                self._streams[gpu_id] = streams

            gpu_name = torch.cuda.get_device_name(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
            logger.info(f"🎮 GPU {gpu_id} 초기화: {gpu_name} ({total_memory:.1f} GB, {self.config.streams_per_gpu} streams)")

        total_streams = sum(len(s) for s in self._streams.values())
        logger.info(f"✅ 총 {total_streams} Stream on {len(gpu_ids)} GPU(s)")
    
    # =========================================================================
    # VRAM 모니터링
    # =========================================================================
    
    def get_vram_usage(self, gpu_id: int = 0) -> Dict[str, float]:
        """GPU별 VRAM 사용량 조회 (GB)"""
        if not CUDA_AVAILABLE:
            return {"allocated": 0, "reserved": 0, "available": 0, "total": 0}

        try:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
            return {
                "allocated": allocated,
                "reserved": reserved,
                "available": total - reserved,
                "total": total,
            }
        except Exception:
            return {"allocated": 0, "reserved": 0, "available": 0, "total": 0}

    def get_all_vram_usage(self) -> Dict[int, Dict[str, float]]:
        """모든 GPU VRAM 사용량"""
        result = {}
        for gpu_id in self.config.gpu_ids:
            result[gpu_id] = self.get_vram_usage(gpu_id)
        return result

    def check_vram_health(self, gpu_id: int = 0) -> bool:
        """GPU VRAM 상태 확인 (85% 한계)"""
        usage = self.get_vram_usage(gpu_id)
        limit = usage["total"] * self.config.vram_limit_ratio
        if usage["allocated"] > limit:
            logger.warning(f"⚠️ GPU {gpu_id} VRAM 한계 초과: {usage['allocated']:.2f}/{limit:.2f} GB")
            return False
        return True

    def auto_adjust_batch_size(self, gpu_id: int = 0) -> int:
        """VRAM 여유에 따라 배치 크기 자동 조정 (GPU별)"""
        if not CUDA_AVAILABLE:
            return 1
        usage = self.get_vram_usage(gpu_id)
        total = max(usage["total"], 1.0)
        ratio = usage["allocated"] / total
        if ratio < 0.5:
            return 4
        elif ratio < 0.7:
            return 3
        else:
            return 2

    def get_batch_size_by_duration(self, duration_sec: float) -> int:
        """영상 길이별 배치 크기 결정 (모두 GPU 처리)"""
        if duration_sec < 30:
            return 4   # 짧은 영상: batch 4
        elif duration_sec <= 120:
            return 2   # 중간 영상: batch 2
        else:
            return 1   # 긴 영상: GPU에서 1개씩 안전하게 처리

    def get_optimal_fps(self, video_duration_sec: float) -> int:
        """영상 길이에 따른 최적 FPS 결정"""
        if video_duration_sec > self.config.long_video_threshold_sec:
            return self.config.long_video_fps
        return self.config.target_fps
    
    # =========================================================================
    # 배치 처리
    # =========================================================================
    
    def process_batch(
        self,
        video_paths: List[str],
        processor: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        영상 길이별 GPU/CPU 라우팅 + 6-Stream 병렬 처리

        Args:
            video_paths: 비디오 경로 목록
            processor: 처리 함수 (None이면 기본 프로세서 사용)
            output_dir: NPZ 저장 디렉토리 (UnifiedVideoProcessor 사용 시)

        Returns:
            처리 결과 목록
        """
        if processor is None:
            processor = self._default_processor

        start_time = time.time()
        results: List[Dict[str, Any]] = []

        # ── 모든 영상을 GPU로 처리 ──
        # RTMPose/YOLO는 프레임 단위 추론이므로 영상 길이와 무관하게 VRAM 안전.
        # CPU 분기는 GPU 대비 10~50x 느려서 파이프라인 병목 원인이었음.
        all_paths = list(video_paths)

        logger.info(
            f"🎬 배치 처리 시작: 총 {len(all_paths)}개 (전부 GPU 처리)"
        )

        if all_paths:
            batch_size = min(len(all_paths), self.auto_adjust_batch_size())

            for i in range(0, len(all_paths), batch_size):
                batch = all_paths[i:i + batch_size]

                # VRAM 상태 확인 (GPU별)
                for gpu_id in self.config.gpu_ids:
                    if not self.check_vram_health(gpu_id):
                        if CUDA_AVAILABLE:
                            torch.cuda.empty_cache()
                        batch_size = max(1, batch_size - 1)
                        break

                batch_results = self._process_batch_parallel(batch, processor)
                results.extend(batch_results)

                progress = (i + len(batch)) / max(len(all_paths), 1) * 100
                logger.info(f"  GPU 진행률: {i + len(batch)}/{len(all_paths)} ({progress:.1f}%)")

        elapsed = time.time() - start_time

        # 통계 업데이트
        with self._lock:
            self._stats["total_processed"] += len(results)
            self._stats["total_time_sec"] += elapsed

            if CUDA_AVAILABLE and self.config.gpu_ids:
                peak_vram = max(
                    torch.cuda.max_memory_allocated(gid) / 1024 ** 3
                    for gid in self.config.gpu_ids
                    if gid < torch.cuda.device_count()
                )
                self._stats["peak_vram_gb"] = max(self._stats["peak_vram_gb"], peak_vram)

        logger.info(f"✅ 배치 처리 완료: {len(results)}개, {elapsed:.1f}초")

        return results

    def _process_batch_parallel(
        self,
        batch: List[str],
        processor: Callable,
    ) -> List[Dict[str, Any]]:
        """배치 병렬 처리 (Dual-GPU 스트림 분배)"""
        results = []
        num_gpus = len(self.config.gpu_ids)

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for slot, video_path in enumerate(batch):
                # GPU 및 스트림 할당 (round-robin)
                gpu_id = self.config.gpu_ids[slot % num_gpus]
                stream_slot = (slot // num_gpus) % self.config.streams_per_gpu
                future = executor.submit(
                    self._process_single,
                    video_path,
                    gpu_id,
                    stream_slot,
                    processor,
                )
                futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"처리 실패: {e}")

        return results

    def _process_single(
        self,
        video_path: str,
        gpu_id: int,
        stream_slot: int,
        processor: Callable,
    ) -> Optional[Dict[str, Any]]:
        """단일 스트림에서 처리 (GPU별 스트림 사용)"""
        try:
            gpu_streams = self._streams.get(gpu_id, [])
            if CUDA_AVAILABLE and stream_slot < len(gpu_streams):
                with torch.cuda.device(gpu_id):
                    with cuda.stream(gpu_streams[stream_slot]):
                        return processor(video_path)
            else:
                return processor(video_path)
        except Exception as e:
            logger.error(f"GPU {gpu_id} 스트림 {stream_slot} 처리 실패: {e}")
            return None
    
    def _default_processor(self, video_path: str) -> Dict[str, Any]:
        """기본 프로세서 (MediaPipe 포즈 추출 + IL 인코딩)"""
        try:
            from scripts.pipeline.build_imitation_data import extract_pose_from_video, encode_imitation_data
            
            # 영상 길이 확인
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # 최적 FPS 결정
            target_fps = self.get_optimal_fps(duration)
            
            # 포즈 추출
            pose_data = extract_pose_from_video(video_path, output_fps=target_fps)
            
            if pose_data and pose_data.get("body") is not None:
                video_id = Path(video_path).stem
                il_data = encode_imitation_data(pose_data, video_id)
                
                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "success": True,
                    "frames": len(pose_data.get("body", [])),
                    "fps": target_fps,
                    "il_data": il_data,
                }
            
            return {
                "video_path": video_path,
                "success": False,
                "error": "포즈 추출 실패",
            }
            
        except Exception as e:
            return {
                "video_path": video_path,
                "success": False,
                "error": str(e),
            }

    def make_detect_processor(self, output_fps: float = 5.0, device: str = None):
        """객체 검출 프로세서 생성 (detect 단계용)"""
        def _detect_processor(video_path: str) -> Dict[str, Any]:
            try:
                from extraction.object_detector import ObjectDetector
                from extraction.detect_to_episodes import _save_episode_npz
                
                detector = ObjectDetector(device=device)
                detections = detector.detect_video(
                    video_path=video_path,
                    output_fps=output_fps,
                    max_frames=None,
                    visualize=False,
                )
                
                video_id = Path(video_path).stem
                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "success": True,
                    "frames": len(detections),
                    "detections": detections,
                }
            except Exception as e:
                return {
                    "video_path": video_path,
                    "success": False,
                    "error": str(e),
                }
        return _detect_processor

    def make_il_processor(self, output_fps: float = 5.0, max_frames: int = None,
                          output_dir: str = None):
        """모방학습 데이터 생성 프로세서 (build_il 단계용) — 결과를 npz로 저장"""
        import numpy as np

        def _il_processor(video_path: str) -> Dict[str, Any]:
            try:
                from scripts.pipeline.build_imitation_data import extract_pose_from_video, encode_imitation_data
                
                video_id = Path(video_path).stem
                
                # 이미 존재하면 스킵
                if output_dir:
                    out_path = Path(output_dir) / f"{video_id}_episode.npz"
                    if out_path.exists():
                        try:
                            d = np.load(str(out_path), allow_pickle=True)
                            if "states" in d and "actions" in d:
                                return {
                                    "video_path": video_path,
                                    "video_id": video_id,
                                    "status": "skipped",
                                    "success": True,
                                    "msg": "already has IL data",
                                }
                        except Exception:
                            pass
                
                # 포즈 추출
                pose_data = extract_pose_from_video(video_path, output_fps=output_fps,
                                                     max_frames=max_frames)
                if pose_data is None:
                    return {
                        "video_path": video_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": "no pose detected",
                    }
                
                T = pose_data["body"].shape[0]
                if T < 3:
                    return {
                        "video_path": video_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": f"too few frames: {T}",
                    }
                
                # IL 인코딩
                il_data = encode_imitation_data(pose_data, video_id)
                
                # 저장
                if output_dir:
                    out_path = Path(output_dir) / f"{video_id}_episode.npz"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_path, **il_data)
                
                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "success": True,
                    "status": "success",
                    "frames": int(T),
                    "state_dim": int(il_data["state_dim"]),
                    "action_dim": int(il_data["action_dim"]),
                }
            except Exception as e:
                return {
                    "video_path": video_path,
                    "video_id": Path(video_path).stem,
                    "success": False,
                    "status": "failed",
                    "error": str(e),
                }
        return _il_processor

    def make_pose_extract_processor(self, output_fps: float = 30.0, max_frames: int = None,
                                     output_dir: str = None):
        """포즈 추출 프로세서 (extract_poses 단계용) — 결과를 npz로 저장"""
        def _pose_processor(video_path: str) -> Dict[str, Any]:
            try:
                from extraction.pose_estimator import MediaPipePoseEstimator
                from extraction.pose_serializer import PoseSerializer

                video_id = Path(video_path).stem
                out_path = None
                if output_dir:
                    out_path = Path(output_dir) / f"{video_id}_pose.npz"
                    if out_path.exists():
                        return {
                            "video_path": video_path,
                            "video_id": video_id,
                            "success": True,
                            "status": "skipped",
                            "msg": "already extracted",
                        }

                estimator = MediaPipePoseEstimator(
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    enable_hands=True,
                )

                sequence = estimator.process_video(
                    video_path,
                    output_fps=output_fps,
                    max_frames=max_frames,
                )

                if not sequence or not sequence.frames:
                    return {
                        "video_path": video_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": "no frames extracted",
                    }

                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    serializer = PoseSerializer()
                    serializer.save_numpy(sequence, out_path)

                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "success": True,
                    "status": "success",
                    "frames": len(sequence.frames),
                    "output_path": str(out_path) if out_path else None,
                }
            except Exception as e:
                return {
                    "video_path": video_path,
                    "video_id": Path(video_path).stem,
                    "success": False,
                    "status": "failed",
                    "error": str(e),
                }
        return _pose_processor

    def make_encode_processor(self, poses_dir: str = None, output_dir: str = None):
        """액션 인코딩 프로세서 (encode_actions 단계용) — 포즈 npz → 에피소드 npz"""
        def _encode_processor(pose_file_path: str) -> Dict[str, Any]:
            try:
                from transformation.encoding import StateBuilder, ActionComputer
                from transformation.spec import TransformConfig, StateSpec, ActionSpec
                import numpy as np

                file_path = Path(pose_file_path)
                video_id = file_path.stem.replace("_pose", "")

                # 이미 에피소드가 있으면 스킵
                if output_dir:
                    ep_path = Path(output_dir) / f"{video_id}_episode.npz"
                    if ep_path.exists():
                        return {
                            "file_path": pose_file_path,
                            "video_id": video_id,
                            "success": True,
                            "status": "skipped",
                            "msg": "already encoded",
                        }

                # 포즈 데이터 로드
                data = np.load(file_path, allow_pickle=True)
                poses = None
                for key in ["poses", "body", "keypoints"]:
                    if key in data:
                        poses = data[key]
                        break
                if poses is None:
                    return {
                        "file_path": pose_file_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": "no pose data found",
                    }

                fps = float(data.get("fps", 30.0))
                confidence = data.get("confidence", data.get("confidences", None))
                T = poses.shape[0]
                if T < 2:
                    return {
                        "file_path": pose_file_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": f"too few frames: {T}",
                    }

                config = TransformConfig()
                state_builder = StateBuilder(
                    config=config,
                    state_spec=StateSpec(
                        joint_positions=True,
                        joint_velocities=True,
                        object_relations=False,
                        confidence_stats=True,
                    ),
                )
                action_computer = ActionComputer(
                    config=config,
                    action_spec=ActionSpec(
                        position_delta=True,
                        rotation_delta=False,
                        gripper_state=False,
                        eef_only=True,
                    ),
                )

                # 포즈 정규화
                if poses.shape[1] >= 25:
                    hip_center = (poses[:, 23, :] + poses[:, 24, :]) / 2
                else:
                    hip_center = np.mean(poses, axis=1)
                normalized = poses - hip_center[:, np.newaxis, :]
                if poses.shape[1] >= 12:
                    sw = np.linalg.norm(poses[:, 11, :] - poses[:, 12, :], axis=1)
                    scale = np.clip(sw, 0.1, 2.0)
                    normalized = normalized / scale[:, np.newaxis, np.newaxis]

                # 속도
                dt = 1.0 / fps
                velocity = np.zeros_like(normalized)
                if T > 2:
                    velocity[1:-1] = (normalized[2:] - normalized[:-2]) / (2 * dt)
                    velocity[0] = (normalized[1] - normalized[0]) / dt
                    velocity[-1] = (normalized[-1] - normalized[-2]) / dt

                states, state_masks = state_builder.build_state(
                    pose=normalized, velocity=velocity, conf=confidence,
                )
                actions, action_masks = action_computer.compute_action(
                    pose=normalized, dt=dt,
                )

                # 저장
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    ep_path = Path(output_dir) / f"{video_id}_episode.npz"
                    np.savez_compressed(ep_path,
                        states=states.astype(np.float32),
                        actions=actions.astype(np.float32),
                        state_masks=state_masks,
                        action_masks=action_masks,
                        poses=normalized.astype(np.float32),
                        velocity=velocity.astype(np.float32),
                        timestamps=data.get("timestamps", np.arange(T) / fps).astype(np.float32),
                        fps=fps,
                        video_id=video_id,
                        state_dim=states.shape[1],
                        action_dim=actions.shape[1],
                        num_frames=T,
                    )

                return {
                    "file_path": pose_file_path,
                    "video_id": video_id,
                    "success": True,
                    "status": "success",
                    "frames": int(T),
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                }
            except Exception as e:
                return {
                    "file_path": pose_file_path,
                    "video_id": Path(pose_file_path).stem.replace("_pose", ""),
                    "success": False,
                    "status": "failed",
                    "error": str(e),
                }
        return _encode_processor

    def make_quality_filter_processor(self, filtered_dir: str = None,
                                       threshold: float = None):
        """품질 필터 프로세서 (filter_quality 단계용)"""
        def _quality_processor(pose_file_path: str) -> Dict[str, Any]:
            try:
                from core.quality_metrics import QualityMetricCalculator, QualityThresholds
                import numpy as np

                file_path = Path(pose_file_path)
                video_id = file_path.stem.replace("_pose", "")

                data = np.load(file_path, allow_pickle=True)
                poses = None
                for key in ["poses", "body", "keypoints"]:
                    if key in data:
                        poses = data[key]
                        break
                if poses is None:
                    return {
                        "file_path": pose_file_path,
                        "video_id": video_id,
                        "success": False,
                        "status": "failed",
                        "error": "no pose data",
                    }

                # visibility 추가 (필요시)
                if len(poses.shape) == 3 and poses.shape[2] == 3:
                    vis = np.ones((poses.shape[0], poses.shape[1]), dtype=np.float32)
                    poses = np.concatenate([poses, vis[:, :, np.newaxis]], axis=2)

                thresholds = QualityThresholds()
                calculator = QualityMetricCalculator(thresholds)
                metrics = calculator.compute(poses)
                passed = metrics.quality_score >= (threshold or 50.0)

                # 통과하면 filtered_dir로 복사
                if passed and filtered_dir:
                    import shutil
                    Path(filtered_dir).mkdir(parents=True, exist_ok=True)
                    dst = Path(filtered_dir) / file_path.name
                    if not dst.exists():
                        shutil.copy2(file_path, dst)

                return {
                    "file_path": pose_file_path,
                    "video_id": video_id,
                    "success": True,
                    "status": "passed" if passed else "rejected",
                    "passed": passed,
                    "quality_score": float(metrics.quality_score),
                    "confidence_mean": float(metrics.confidence_mean),
                    "jitter_score": float(metrics.jitter_score),
                }
            except Exception as e:
                return {
                    "file_path": pose_file_path,
                    "video_id": Path(pose_file_path).stem.replace("_pose", ""),
                    "success": False,
                    "status": "failed",
                    "error": str(e),
                }
        return _quality_processor

    # =========================================================================
    # 통계
    # =========================================================================
    
    @property
    def stats(self) -> Dict[str, Any]:
        """처리 통계"""
        return self._stats.copy()
    
    def print_stats(self):
        """통계 출력"""
        s = self._stats
        avg_time = s["total_time_sec"] / max(1, s["total_processed"])
        n_gpus = len(self.config.gpu_ids)
        total_streams = n_gpus * self.config.streams_per_gpu

        print(f"""
{'='*60}
📊 GPU {total_streams}-Stream 처리 통계 ({n_gpus} GPU)
{'='*60}
  총 처리: {s['total_processed']}개
  총 시간: {s['total_time_sec']:.1f}초
  평균 처리 시간: {avg_time:.2f}초/영상
  피크 VRAM: {s['peak_vram_gb']:.2f} GB
""")


def monitor_gpu(interval_sec: float = 1.0, duration_sec: float = 60.0):
    """GPU 모니터링 (별도 스레드용)"""
    if not CUDA_AVAILABLE:
        print("CUDA 사용 불가")
        return
    
    start = time.time()
    
    while time.time() - start < duration_sec:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        # nvidia-smi로 사용률 조회
        try:
            import subprocess
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ])
            util = float(output.decode().strip())
        except:
            util = 0
        
        print(f"GPU: {util:.0f}% | VRAM: {allocated:.2f}/{reserved:.2f} GB")
        time.sleep(interval_sec)


if __name__ == "__main__":
    # 테스트
    manager = GPU3StreamManager()
    
    print("CUDA 사용 가능:", CUDA_AVAILABLE)
    print("VRAM:", manager.get_vram_usage())
    print("권장 배치 크기:", manager.auto_adjust_batch_size())
