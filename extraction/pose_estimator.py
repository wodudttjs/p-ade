"""
MediaPipe 기반 포즈 추정기

비디오에서 전신 및 손 포즈를 추출합니다.
"""

import sys
import os
from pathlib import Path

_vendor_dir = Path(__file__).resolve().parent.parent / "vendor"
_mediapipe_dir = _vendor_dir / "mediapipe"
if _mediapipe_dir.exists() and os.getenv("USE_VENDOR_MEDIAPIPE") == "1":
    sys.path.insert(0, str(_vendor_dir))

import cv2
import mediapipe as mp
import numpy as np
import argparse
from typing import List, Optional, Dict, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
import time
import shutil

from core.logging_config import logger


@dataclass
class PoseLandmark:
    """단일 랜드마크"""
    x: float  # 정규화된 좌표 (0.0 ~ 1.0)
    y: float
    z: float  # 깊이 (상대적)
    visibility: float  # 가시성 (0.0 ~ 1.0)
    
    def to_array(self) -> np.ndarray:
        """NumPy 배열로 변환"""
        return np.array([self.x, self.y, self.z])


@dataclass
class FramePose:
    """단일 프레임의 포즈 데이터"""
    frame_idx: int
    timestamp: float  # 초 단위
    
    # Body pose (33 landmarks)
    body_landmarks: List[PoseLandmark] = field(default_factory=list)
    body_world_landmarks: List[PoseLandmark] = field(default_factory=list)  # 3D 월드 좌표
    
    # Hand poses (21 landmarks each)
    left_hand_landmarks: Optional[List[PoseLandmark]] = None
    right_hand_landmarks: Optional[List[PoseLandmark]] = None
    
    # 메타데이터
    pose_confidence: float = 0.0  # 전체 포즈 신뢰도
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """NumPy 배열로 변환"""
        result = {}
        
        if self.body_landmarks:
            result['body'] = np.array([lm.to_array() for lm in self.body_landmarks])
        
        if self.body_world_landmarks:
            result['body_world'] = np.array([lm.to_array() for lm in self.body_world_landmarks])
        
        if self.left_hand_landmarks:
            result['left_hand'] = np.array([lm.to_array() for lm in self.left_hand_landmarks])
        
        if self.right_hand_landmarks:
            result['right_hand'] = np.array([lm.to_array() for lm in self.right_hand_landmarks])
        
        return result


@dataclass
class VideoPoseSequence:
    """전체 비디오의 포즈 시퀀스"""
    video_path: str
    fps: float
    total_frames: int
    
    frames: List[FramePose] = field(default_factory=list)
    
    def get_as_numpy(self) -> Dict[str, np.ndarray]:
        """
        시계열 NumPy 배열로 변환
        
        Returns:
            {
                'body': [T, 33, 3],
                'body_world': [T, 33, 3],
                'left_hand': [T, 21, 3],
                'right_hand': [T, 21, 3],
                'timestamps': [T],
                'confidence': [T]
            }
        """
        T = len(self.frames)
        
        body_array = np.zeros((T, 33, 3))
        body_world_array = np.zeros((T, 33, 3))
        left_hand_array = np.zeros((T, 21, 3))
        right_hand_array = np.zeros((T, 21, 3))
        timestamps = np.zeros(T)
        confidence = np.zeros(T)
        
        for i, frame_pose in enumerate(self.frames):
            frame_data = frame_pose.to_numpy()
            
            if 'body' in frame_data:
                body_array[i] = frame_data['body']
            if 'body_world' in frame_data:
                body_world_array[i] = frame_data['body_world']
            if 'left_hand' in frame_data:
                left_hand_array[i] = frame_data['left_hand']
            if 'right_hand' in frame_data:
                right_hand_array[i] = frame_data['right_hand']
            
            timestamps[i] = frame_pose.timestamp
            confidence[i] = frame_pose.pose_confidence
        
        return {
            'body': body_array,
            'body_world': body_world_array,
            'left_hand': left_hand_array,
            'right_hand': right_hand_array,
            'timestamps': timestamps,
            'confidence': confidence,
        }


class MediaPipePoseEstimator:
    """MediaPipe 포즈 추정기"""
    
    def __init__(
        self,
        model_complexity: int = 1,  # 0, 1, 2 (높을수록 정확하지만 느림)
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_hands: bool = True,
    ):
        """
        Args:
            model_complexity: 모델 복잡도 (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: 최소 검출 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
            enable_hands: 손 추적 활성화
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_hands = enable_hands
        
        # MediaPipe 초기화
        self._ensure_mediapipe_resources()
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pose 모델
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 비디오 모드
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Hands 모델
        self.hands = None
        if enable_hands:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        
        logger.info(
            f"MediaPipe initialized: complexity={model_complexity}, "
            f"hands={'enabled' if enable_hands else 'disabled'}"
        )

    def _ensure_mediapipe_resources(self) -> None:
        try:
            from mediapipe.python import resource_util
        except Exception:
            return

        package_dir = Path(mp.__file__).resolve().parent
        source_modules = package_dir / "modules"
        target_root = Path(__file__).resolve().parent.parent / "mediapipe_resources"
        target_modules = target_root / "modules"

        if source_modules.exists() and not target_modules.exists():
            shutil.copytree(source_modules, target_modules, dirs_exist_ok=True)

        if target_root.exists():
            resource_util.set_resource_dir(str(target_root))
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> Optional[FramePose]:
        """
        단일 프레임 처리
        
        Args:
            frame: BGR 이미지 (OpenCV 형식)
            frame_idx: 프레임 번호
            timestamp: 타임스탬프 (초)
        
        Returns:
            FramePose or None
        """
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 포즈 추정
        pose_results = self.pose.process(rgb_frame)
        
        if not pose_results.pose_landmarks:
            return None
        
        # FramePose 생성
        frame_pose = FramePose(
            frame_idx=frame_idx,
            timestamp=timestamp,
        )
        
        # Body landmarks
        frame_pose.body_landmarks = self._convert_landmarks(
            pose_results.pose_landmarks.landmark
        )
        
        # World landmarks (3D)
        if pose_results.pose_world_landmarks:
            frame_pose.body_world_landmarks = self._convert_landmarks(
                pose_results.pose_world_landmarks.landmark
            )
        
        # 신뢰도 계산 (평균 visibility)
        visibilities = [lm.visibility for lm in frame_pose.body_landmarks]
        frame_pose.pose_confidence = np.mean(visibilities)
        
        # 손 추정
        if self.hands:
            hands_results = self.hands.process(rgb_frame)
            
            if hands_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    hands_results.multi_hand_landmarks,
                    hands_results.multi_handedness
                ):
                    hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                    landmarks = self._convert_landmarks(hand_landmarks.landmark)
                    
                    if hand_label == 'Left':
                        frame_pose.left_hand_landmarks = landmarks
                    else:
                        frame_pose.right_hand_landmarks = landmarks
        
        return frame_pose
    
    @staticmethod
    def _convert_landmarks(landmarks) -> List[PoseLandmark]:
        """MediaPipe 랜드마크를 PoseLandmark로 변환"""
        result = []
        for lm in landmarks:
            result.append(PoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, 'visibility', 1.0),
            ))
        return result
    
    def process_video(
        self,
        video_path: str,
        output_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> VideoPoseSequence:
        """
        전체 비디오 처리
        
        Args:
            video_path: 비디오 파일 경로
            output_fps: 출력 FPS (None = 원본 FPS)
            max_frames: 최대 프레임 수 (테스트용)
            progress_callback: 진행률 콜백 (frame_idx, total_frames)
        
        Returns:
            VideoPoseSequence
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 비디오 정보
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_fps is None:
            output_fps = original_fps
        
        # 프레임 샘플링 간격
        frame_interval = int(original_fps / output_fps) if output_fps < original_fps else 1
        
        sequence = VideoPoseSequence(
            video_path=video_path,
            fps=output_fps,
            total_frames=total_frames,
        )
        
        frame_idx = 0
        processed_count = 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Original FPS: {original_fps}, Output FPS: {output_fps}")
        logger.info(f"Frame interval: {frame_interval}")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임 샘플링
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # 최대 프레임 수 제한
            if max_frames and processed_count >= max_frames:
                break
            
            # 타임스탬프 계산
            timestamp = frame_idx / original_fps
            
            # 포즈 추정
            frame_pose = self.process_frame(frame, frame_idx, timestamp)
            
            if frame_pose:
                sequence.frames.append(frame_pose)
            
            processed_count += 1
            
            # 진행률 콜백
            if progress_callback and processed_count % 10 == 0:
                progress_callback(processed_count, total_frames // frame_interval)
            
            frame_idx += 1
        
        cap.release()
        
        elapsed = time.time() - start_time
        fps_processed = len(sequence.frames) / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Processed {len(sequence.frames)} frames in {elapsed:.2f}s "
            f"({fps_processed:.2f} fps)"
        )
        
        return sequence
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        frame_pose: FramePose,
        draw_body: bool = True,
        draw_hands: bool = True,
    ) -> np.ndarray:
        """
        프레임에 포즈 시각화
        
        Args:
            frame: 원본 프레임 (BGR)
            frame_pose: 포즈 데이터
            draw_body: 신체 포즈 그리기 여부
            draw_hands: 손 포즈 그리기 여부
        
        Returns:
            시각화된 프레임
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 신체 포즈 그리기
        if draw_body and frame_pose.body_landmarks:
            self._draw_pose_landmarks(
                vis_frame,
                frame_pose.body_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # 왼손 그리기
        if draw_hands and frame_pose.left_hand_landmarks:
            self._draw_hand(
                vis_frame,
                frame_pose.left_hand_landmarks,
                (0, 255, 0)  # 녹색
            )
        
        # 오른손 그리기
        if draw_hands and frame_pose.right_hand_landmarks:
            self._draw_hand(
                vis_frame,
                frame_pose.right_hand_landmarks,
                (255, 0, 0)  # 파란색
            )
        
        return vis_frame
    
    def _draw_pose_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[PoseLandmark],
        connections
    ):
        """신체 랜드마크 그리기"""
        h, w = image.shape[:2]
        
        # 연결선 그리기
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # 가시성 체크
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue
            
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)
        
        # 관절점 그리기
        for lm in landmarks:
            if lm.visibility < 0.5:
                continue
            
            point = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, point, 3, (0, 0, 255), -1)
    
    def _draw_hand(
        self,
        image: np.ndarray,
        hand_landmarks: List[PoseLandmark],
        color: tuple
    ):
        """손 랜드마크 그리기"""
        h, w = image.shape[:2]
        
        # 연결선
        connections = self.mp_hands.HAND_CONNECTIONS
        
        for connection in connections:
            start_idx, end_idx = connection
            
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]
            
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            
            cv2.line(image, start_point, end_point, color, 2)
        
        # 관절점
        for lm in hand_landmarks:
            point = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, point, 3, color, -1)
    
    def save_sequence(self, sequence: 'VideoPoseSequence', output_path: str):
        """포즈 시퀀스를 파일로 저장 (.npz)"""
        data = sequence.get_as_numpy()
        
        np.savez_compressed(
            output_path,
            video_path=sequence.video_path,
            fps=sequence.fps,
            total_frames=sequence.total_frames,
            **data
        )
        
        logger.info(f"Saved pose sequence to {output_path}")
    
    @staticmethod
    def load_sequence(filepath: str) -> Dict[str, np.ndarray]:
        """저장된 포즈 시퀀스 로드"""
        data = np.load(filepath, allow_pickle=True)
        return dict(data)
    
    def close(self):
        """리소스 해제"""
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
        logger.info("MediaPipe resources released")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _iter_videos(video_path: Optional[str], input_dir: Optional[str]) -> Iterable[Path]:
    if video_path:
        path = Path(video_path)
        if path.exists() and path.is_file():
            yield path
        return
    if input_dir:
        root = Path(input_dir)
        if root.exists():
            for p in root.glob("*.mp4"):
                if p.is_file():
                    yield p


def _get_db_session(db_path: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models.database import Base

    db_path_obj = Path(db_path)
    if not db_path_obj.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        db_path_obj = project_root / db_path_obj

    engine = create_engine(f"sqlite:///{db_path_obj}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _update_db_after_pose(session, video_path: Path, output_path: Path, sequence: VideoPoseSequence):
    from models.database import Video, Episode

    video_id = video_path.stem
    video = session.query(Video).filter_by(video_id=video_id).first()
    if not video:
        video = Video(
            video_id=video_id,
            platform="youtube",
            url="",
            status="processed",
        )
        session.add(video)

    video.processed_at = datetime.utcnow()
    video.local_path = str(video_path)

    episode_id = f"{video_id}_ep001"
    episode = session.query(Episode).filter_by(episode_id=episode_id).first()
    if not episode:
        episode = Episode(
            video_id=video.id,
            episode_id=episode_id,
        )
        session.add(episode)

    episode.start_frame = 0
    episode.end_frame = sequence.total_frames
    episode.duration_frames = sequence.total_frames
    episode.local_path = str(output_path)
    episode.filesize_bytes = output_path.stat().st_size if output_path.exists() else None

    if sequence.frames:
        confidences = [f.pose_confidence for f in sequence.frames]
        episode.confidence_score = float(np.mean(confidences))
        episode.quality_score = float(np.mean(confidences))
        episode.jittering_score = 0.0


def run_cli():
    parser = argparse.ArgumentParser(description="MediaPipe 포즈 추출")
    parser.add_argument("--video", help="단일 비디오 경로")
    parser.add_argument("--input-dir", default="data/raw", help="비디오 폴더 경로")
    parser.add_argument("--output-dir", default="data/episodes", help="출력 폴더 경로")
    parser.add_argument("--fps", type=float, default=None, help="출력 FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="최대 프레임 수")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _get_db_session(args.db)
    processed = 0

    try:
        with MediaPipePoseEstimator() as estimator:
            for video in _iter_videos(args.video, args.input_dir):
                logger.info(f"Processing video: {video}")
                sequence = estimator.process_video(
                    video_path=str(video),
                    output_fps=args.fps,
                    max_frames=args.max_frames,
                )
                if not sequence:
                    logger.warning(f"No pose detected: {video}")
                    continue

                output_path = output_dir / f"{video.stem}_episode.npz"
                estimator.save_sequence(sequence, str(output_path))
                _update_db_after_pose(session, video, output_path, sequence)
                session.commit()
                processed += 1

    finally:
        session.close()

    print(f"✅ 포즈 추출 완료: {processed}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
