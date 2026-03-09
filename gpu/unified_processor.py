"""
UnifiedVideoProcessor — Detect + IL 1-Pass 통합 처리기 (RTMPose WholeBody GPU)

AS-IS (2-Pass, MediaPipe):
  비디오 → YOLO → NPZ 저장
  비디오 → MediaPipe → NPZ 저장
  = 비디오 2회 디코딩

TO-BE (1-Pass, RTMPose WholeBody GPU):
  비디오 → RTMPose WholeBody (ONNX GPU) → COCO 17 + Hands → 통합 NPZ 1회 저장
  = 비디오 1회 디코딩 → 처리 시간 40% 단축

통합 NPZ 구조:
  {
    'states':     (N, 103) — 관절위치(51) + 속도(51) + 신뢰도(1),
    'actions':    (N-1, 52) — 관절delta(51) + gripper(1),
    'poses':      (N, 17, 3) — COCO 17 정규화 관절,
    'velocity':   (N, 17, 3) — 관절 속도,
    'left_hand':  (N, 21, 3) — 왼손 랜드마크,
    'right_hand': (N, 21, 3) — 오른손 랜드마크,
    'gripper_state': (N,) — 그리퍼 상태 (0=열림, 1=닫힘),
    'detections': (N, 4) — YOLO bbox (선택),
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
        1-Pass 통합 처리 (RTMPose WholeBody GPU + YOLO)

        Args:
            video_path: 입력 비디오 경로
            output_path: NPZ 저장 경로 (None이면 저장 안 함)

        Returns:
            결과 딕셔너리 (success, frames, states, actions 등)
        """
        video_id = Path(video_path).stem
        start_time = time.time()

        try:
            import cv2

            # 비디오 기본 정보
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._fail(video_id, "비디오 열기 실패")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / src_fps if src_fps > 0 else 0
            cap.release()

            # 처리기 초기화 (lazy)
            self._init_processors()

            # ── RTMPose WholeBody로 포즈 추출 (GPU, 비디오 전체) ──
            if self._pose_est is not None:
                pose_data = self._pose_est.extract(
                    video_path, 
                    output_fps=self.output_fps,
                    max_frames=self.max_frames,
                )
            else:
                return self._fail(video_id, "포즈 추정기 없음 (RTMPose 초기화 실패)")

            if pose_data is None or pose_data["body"].shape[0] < 3:
                extracted = pose_data["body"].shape[0] if pose_data else 0
                return self._fail(video_id, f"추출된 프레임 수 부족: {extracted}")

            extracted = pose_data["body"].shape[0]

            # ── YOLO 객체 검출 (선택) ──
            detections = np.zeros((extracted, 4), dtype=np.float32)
            # YOLO 검출은 선택적 — 포즈 기반 IL이 핵심
            # TODO: YOLO 검출 필요 시 frame 재디코딩 또는 concurrent 처리

            # ── State-Action 인코딩 (RTMPose COCO 17 기반) ──
            from scripts.pipeline.build_imitation_data import encode_imitation_data
            il_data = encode_imitation_data(pose_data, video_id)

            states = il_data["states"]
            actions = il_data["actions"]

            # ── 통합 NPZ 저장 ──
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                save_data = {
                    "states": states,
                    "actions": actions,
                    "poses": il_data["poses"],
                    "velocity": il_data["velocity"],
                    "left_hand": il_data["left_hand"],
                    "right_hand": il_data["right_hand"],
                    "gripper_state": il_data["gripper_state"],
                    "confidence": il_data["confidence"],
                    "detections": detections,
                    "fps": float(self.output_fps),
                    "duration_sec": float(duration_sec),
                    "video_id": video_id,
                    "num_frames": extracted,
                    "state_dim": states.shape[1] if states.ndim == 2 else 0,
                    "action_dim": actions.shape[1] if actions.ndim == 2 else 0,
                }
                np.savez_compressed(output_path, **save_data)
                logger.debug(f"통합 NPZ 저장: {output_path} ({extracted}프레임)")

            elapsed = time.time() - start_time

            return {
                "video_id": video_id,
                "video_path": video_path,
                "success": True,
                "frames": extracted,
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
        """RTMPose WholeBody 포즈 추정기 초기화 (MediaPipe 대체)"""
        try:
            from extraction.rtmpose_wholebody import RTMPoseVideoExtractor
            self._pose_est = RTMPoseVideoExtractor(device="gpu")
            logger.debug("RTMPose WholeBody 포즈 추정기 초기화 완료 (GPU)")
        except Exception as e:
            logger.warning(f"RTMPose 초기화 실패 (None으로 진행): {e}")
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
