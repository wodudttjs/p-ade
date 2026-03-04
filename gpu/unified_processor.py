"""
UnifiedVideoProcessor — Detect + IL 1-Pass 통합 처리기

AS-IS (2-Pass):
  비디오 → YOLO → NPZ 저장
  비디오 → MediaPipe → NPZ 저장
  = 비디오 2회 디코딩

TO-BE (1-Pass):
  비디오 → (YOLO + MediaPipe) → 통합 NPZ 1회 저장
  = 비디오 1회 디코딩 → 처리 시간 40% 단축 목표

통합 NPZ 구조:
  {
    'detections': (N, D) — YOLO 결과,
    'poses':      (N, 33, 3) — MediaPipe body,
    'states':     (N, D_s) — State 인코딩,
    'actions':    (N-1, D_a) — Action 인코딩,
    'metadata':   dict
  }
"""

import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)

# CUDA 선택적 임포트
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    torch = None


class UnifiedVideoProcessor:
    """
    1-Pass 통합 비디오 처리기

    단일 비디오 디코딩으로 YOLO 객체 검출 + MediaPipe 포즈 추출 +
    State-Action 인코딩을 동시에 수행합니다.

    사용법:
        processor = UnifiedVideoProcessor(output_fps=5.0, device="cuda:0")
        result = processor.process(video_path, output_path)
    """

    def __init__(
        self,
        output_fps: float = 5.0,
        device: Optional[str] = None,
        max_frames: Optional[int] = None,
        yolo_confidence: float = 0.3,
    ):
        """
        Args:
            output_fps: 추출 FPS (기본 5.0)
            device: CUDA 디바이스 (None=자동 감지)
            max_frames: 영상당 최대 프레임 수 (None=제한 없음)
            yolo_confidence: YOLO 객체 검출 최소 신뢰도
        """
        self.output_fps = output_fps
        self.device = device or ("cuda:0" if CUDA_AVAILABLE else "cpu")
        self.max_frames = max_frames
        self.yolo_confidence = yolo_confidence

        self._detector = None   # lazy init
        self._pose_est = None   # lazy init
        self._init_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────

    def process(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        1-Pass 통합 처리

        Args:
            video_path: 입력 비디오 경로
            output_path: NPZ 저장 경로 (None이면 저장 안 함)

        Returns:
            {
                "video_id": str,
                "success": bool,
                "frames": int,
                "detections": np.ndarray,
                "poses": np.ndarray,
                "states": np.ndarray,
                "actions": np.ndarray,
                "elapsed_sec": float,
                "error": str (실패 시),
            }
        """
        video_id = Path(video_path).stem
        start_time = time.time()

        try:
            import cv2

            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._fail(video_id, "비디오 열기 실패")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / src_fps if src_fps > 0 else 0

            # 몇 프레임마다 추출할지 계산
            frame_interval = max(1, int(round(src_fps / self.output_fps)))

            # 처리기 초기화 (lazy)
            self._init_processors()

            # ── 1-Pass 프레임 순회 ──
            detections_list = []
            poses_list = []
            confidences_list = []
            frame_idx = 0
            extracted = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_data = self._process_frame(frame, frame_idx)
                    detections_list.append(frame_data["detection"])
                    poses_list.append(frame_data["pose"])
                    confidences_list.append(frame_data["confidence"])
                    extracted += 1

                    if self.max_frames and extracted >= self.max_frames:
                        break

                frame_idx += 1

            cap.release()

            if extracted < 3:
                return self._fail(video_id, f"추출된 프레임 수 부족: {extracted}")

            # ── 배열 변환 ──
            poses = np.array(poses_list, dtype=np.float32)          # (N, 33, 3)
            detections = np.array(detections_list, dtype=np.float32) # (N, D)
            confidences = np.array(confidences_list, dtype=np.float32) # (N,)

            # ── State-Action 인코딩 ──
            states, actions = self._encode_state_action(poses, self.output_fps)

            # ── 통합 NPZ 저장 ──
            if output_path:
                self._save_unified_npz(
                    output_path=output_path,
                    video_id=video_id,
                    detections=detections,
                    poses=poses,
                    states=states,
                    actions=actions,
                    confidences=confidences,
                    fps=self.output_fps,
                    duration_sec=duration_sec,
                )

            elapsed = time.time() - start_time

            return {
                "video_id": video_id,
                "video_path": video_path,
                "success": True,
                "frames": extracted,
                "detections": detections,
                "poses": poses,
                "states": states,
                "actions": actions,
                "state_dim": states.shape[1] if states.ndim == 2 else 0,
                "action_dim": actions.shape[1] if actions.ndim == 2 else 0,
                "elapsed_sec": elapsed,
                "duration_sec": duration_sec,
            }

        except Exception as e:
            logger.error(f"UnifiedProcessor 실패 [{video_id}]: {e}")
            return self._fail(video_id, str(e))

    def process_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        여러 영상 순차 처리 (GPU 스트림 관리는 StreamManager에서 담당)

        Args:
            video_paths: 비디오 경로 목록
            output_dir: NPZ 저장 디렉토리 (None이면 저장 안 함)
        """
        results = []
        total = len(video_paths)

        for i, vp in enumerate(video_paths, 1):
            video_id = Path(vp).stem
            out_path = None
            if output_dir:
                out_path = str(Path(output_dir) / f"{video_id}_episode.npz")
                # 이미 존재하면 스킵
                if Path(out_path).exists():
                    results.append({
                        "video_id": video_id,
                        "video_path": vp,
                        "success": True,
                        "status": "skipped",
                        "frames": 0,
                    })
                    continue

            logger.info(f"  [{i}/{total}] 처리 중: {video_id}")
            result = self.process(vp, out_path)
            result.setdefault("status", "success" if result["success"] else "failed")
            results.append(result)

        return results

    # ──────────────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────────────

    def _init_processors(self):
        """YOLO / MediaPipe 지연 초기화 (스레드 안전)"""
        with self._init_lock:
            if self._pose_est is None:
                self._init_pose_estimator()
            if self._detector is None:
                self._init_detector()

    def _init_pose_estimator(self):
        """MediaPipe 포즈 추정기 초기화"""
        try:
            from extraction.pose_estimator import MediaPipePoseEstimator
            self._pose_est = MediaPipePoseEstimator(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                enable_hands=True,
            )
            logger.debug("MediaPipe 포즈 추정기 초기화 완료")
        except Exception as e:
            logger.warning(f"MediaPipe 초기화 실패 (None으로 진행): {e}")
            self._pose_est = None

    def _init_detector(self):
        """YOLO 객체 검출기 초기화"""
        try:
            from extraction.object_detector import ObjectDetector
            self._detector = ObjectDetector(device=self.device)
            logger.debug(f"YOLO 검출기 초기화 완료 (device={self.device})")
        except Exception as e:
            logger.warning(f"YOLO 초기화 실패 (None으로 진행): {e}")
            self._detector = None

    def _process_frame(self, frame, frame_idx: int) -> Dict[str, Any]:
        """단일 프레임 처리: YOLO + MediaPipe 동시 추론"""
        import cv2

        # ── YOLO 객체 검출 ──
        detection = np.zeros(4, dtype=np.float32)  # [x, y, w, h]
        if self._detector is not None:
            try:
                dets = self._detector.detect_frame(frame, confidence=self.yolo_confidence)
                if dets:
                    # 첫 번째 검출 결과 사용
                    d = dets[0]
                    detection = np.array([
                        d.get("x", 0.0),
                        d.get("y", 0.0),
                        d.get("w", 0.0),
                        d.get("h", 0.0),
                    ], dtype=np.float32)
            except Exception:
                pass

        # ── MediaPipe 포즈 추출 ──
        pose = np.zeros((33, 3), dtype=np.float32)
        confidence = 0.0

        if self._pose_est is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = self._pose_est.process_frame(rgb)
                if pose_result is not None:
                    landmarks = pose_result.get("body")
                    if landmarks is not None and len(landmarks) == 33:
                        pose = np.array(landmarks, dtype=np.float32)
                    confidence = float(pose_result.get("confidence", 0.0))
            except Exception:
                pass

        return {"detection": detection, "pose": pose, "confidence": confidence}

    def _encode_state_action(
        self,
        poses: np.ndarray,
        fps: float,
    ):
        """포즈 배열에서 State / Action 인코딩"""
        try:
            from scripts.pipeline.build_imitation_data import encode_imitation_data
            il_data = encode_imitation_data(
                {"body": poses, "left_hand": None, "right_hand": None},
                video_id="",
            )
            return il_data.get("states", np.zeros((len(poses), 1), dtype=np.float32)), \
                   il_data.get("actions", np.zeros((max(0, len(poses) - 1), 1), dtype=np.float32))
        except Exception:
            pass

        # 폴백: 간단한 포즈 기반 State-Action
        T = len(poses)
        dt = 1.0 / max(fps, 1.0)

        # State: 포즈 좌표 평탄화
        poses_flat = poses.reshape(T, -1)  # (T, 33*3)

        # Action: 연속 프레임 간 차분
        if T > 1:
            actions = np.diff(poses_flat, axis=0).astype(np.float32)  # (T-1, 99)
        else:
            actions = np.zeros((0, poses_flat.shape[1]), dtype=np.float32)

        return poses_flat.astype(np.float32), actions

    def _save_unified_npz(
        self,
        output_path: str,
        video_id: str,
        detections: np.ndarray,
        poses: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        confidences: np.ndarray,
        fps: float,
        duration_sec: float,
    ):
        """통합 NPZ 저장"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        T = len(poses)
        np.savez_compressed(
            output_path,
            detections=detections,
            poses=poses,
            states=states,
            actions=actions,
            confidence=confidences,
            fps=float(fps),
            duration_sec=float(duration_sec),
            video_id=video_id,
            num_frames=T,
            state_dim=states.shape[1] if states.ndim == 2 else 0,
            action_dim=actions.shape[1] if actions.ndim == 2 else 0,
        )
        logger.debug(f"통합 NPZ 저장: {output_path} ({T}프레임)")

    @staticmethod
    def _fail(video_id: str, error: str) -> Dict[str, Any]:
        return {
            "video_id": video_id,
            "success": False,
            "status": "failed",
            "frames": 0,
            "error": error,
            "elapsed_sec": 0.0,
        }
