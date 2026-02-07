"""
포즈 추정기 테스트
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from extraction.pose_estimator import (
    PoseLandmark,
    FramePose,
    VideoPoseSequence,
    MediaPipePoseEstimator
)


def test_pose_landmark_creation():
    """PoseLandmark 생성"""
    landmark = PoseLandmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
    
    assert landmark.x == 0.5
    assert landmark.y == 0.3
    assert landmark.z == 0.1
    assert landmark.visibility == 0.9


def test_pose_landmark_to_array():
    """PoseLandmark 배열 변환"""
    landmark = PoseLandmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
    array = landmark.to_array()
    
    assert isinstance(array, np.ndarray)
    assert array.shape == (3,)
    assert np.allclose(array, [0.5, 0.3, 0.1])


def test_frame_pose_creation():
    """FramePose 생성"""
    frame_pose = FramePose(
        frame_idx=10,
        timestamp=0.33,
    )
    
    assert frame_pose.frame_idx == 10
    assert frame_pose.timestamp == 0.33
    assert frame_pose.body_landmarks == []
    assert frame_pose.pose_confidence == 0.0


def test_frame_pose_with_landmarks():
    """랜드마크가 있는 FramePose"""
    landmarks = [
        PoseLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0),
        PoseLandmark(x=0.6, y=0.4, z=0.1, visibility=0.9),
    ]
    
    frame_pose = FramePose(
        frame_idx=0,
        timestamp=0.0,
        body_landmarks=landmarks,
        pose_confidence=0.95
    )
    
    assert len(frame_pose.body_landmarks) == 2
    assert frame_pose.pose_confidence == 0.95


def test_frame_pose_to_numpy():
    """FramePose NumPy 변환"""
    body_landmarks = [
        PoseLandmark(x=0.1, y=0.2, z=0.3, visibility=1.0) for _ in range(33)
    ]
    
    frame_pose = FramePose(
        frame_idx=0,
        timestamp=0.0,
        body_landmarks=body_landmarks
    )
    
    numpy_data = frame_pose.to_numpy()
    
    assert 'body' in numpy_data
    assert numpy_data['body'].shape == (33, 3)


def test_frame_pose_to_numpy_with_hands():
    """손 포함 NumPy 변환"""
    body_landmarks = [PoseLandmark(x=0, y=0, z=0, visibility=1) for _ in range(33)]
    hand_landmarks = [PoseLandmark(x=0, y=0, z=0, visibility=1) for _ in range(21)]
    
    frame_pose = FramePose(
        frame_idx=0,
        timestamp=0.0,
        body_landmarks=body_landmarks,
        left_hand_landmarks=hand_landmarks,
        right_hand_landmarks=hand_landmarks
    )
    
    numpy_data = frame_pose.to_numpy()
    
    assert 'body' in numpy_data
    assert 'left_hand' in numpy_data
    assert 'right_hand' in numpy_data
    assert numpy_data['left_hand'].shape == (21, 3)
    assert numpy_data['right_hand'].shape == (21, 3)


def test_video_pose_sequence_creation():
    """VideoPoseSequence 생성"""
    sequence = VideoPoseSequence(
        video_path="/path/to/video.mp4",
        fps=30.0,
        total_frames=900
    )
    
    assert sequence.video_path == "/path/to/video.mp4"
    assert sequence.fps == 30.0
    assert sequence.total_frames == 900
    assert len(sequence.frames) == 0


def test_video_pose_sequence_get_as_numpy():
    """VideoPoseSequence NumPy 변환"""
    sequence = VideoPoseSequence(
        video_path="test.mp4",
        fps=30.0,
        total_frames=10
    )
    
    # 샘플 프레임 추가
    for i in range(10):
        body_landmarks = [PoseLandmark(x=0, y=0, z=0, visibility=1) for _ in range(33)]
        frame_pose = FramePose(
            frame_idx=i,
            timestamp=i / 30.0,
            body_landmarks=body_landmarks,
            pose_confidence=0.9
        )
        sequence.frames.append(frame_pose)
    
    numpy_data = sequence.get_as_numpy()
    
    assert 'body' in numpy_data
    assert 'timestamps' in numpy_data
    assert 'confidence' in numpy_data
    assert numpy_data['body'].shape == (10, 33, 3)
    assert numpy_data['timestamps'].shape == (10,)
    assert numpy_data['confidence'].shape == (10,)


def test_mediapipe_estimator_initialization():
    """MediaPipePoseEstimator 초기화"""
    estimator = MediaPipePoseEstimator(
        model_complexity=1,
        min_detection_confidence=0.5,
        enable_hands=True
    )
    
    assert estimator.model_complexity == 1
    assert estimator.min_detection_confidence == 0.5
    assert estimator.enable_hands is True
    assert estimator.pose is not None
    assert estimator.hands is not None
    
    estimator.close()


def test_mediapipe_estimator_without_hands():
    """손 추적 비활성화"""
    estimator = MediaPipePoseEstimator(enable_hands=False)
    
    assert estimator.hands is None
    
    estimator.close()


def test_convert_landmarks():
    """랜드마크 변환"""
    # Mock MediaPipe 랜드마크
    mock_landmarks = []
    for i in range(5):
        mock_lm = Mock()
        mock_lm.x = float(i) * 0.1
        mock_lm.y = float(i) * 0.2
        mock_lm.z = float(i) * 0.3
        mock_lm.visibility = 1.0
        mock_landmarks.append(mock_lm)
    
    converted = MediaPipePoseEstimator._convert_landmarks(mock_landmarks)
    
    assert len(converted) == 5
    assert all(isinstance(lm, PoseLandmark) for lm in converted)
    assert converted[0].x == 0.0
    assert converted[1].x == 0.1


@patch('extraction.pose_estimator.cv2.VideoCapture')
@patch('extraction.pose_estimator.cv2.cvtColor')
def test_process_frame_no_pose(mock_cvtcolor, mock_video_cap):
    """포즈 감지 실패"""
    estimator = MediaPipePoseEstimator()
    
    # Mock 프레임
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = frame
    
    # Mock 포즈 결과 (포즈 없음)
    with patch.object(estimator.pose, 'process') as mock_process:
        mock_result = Mock()
        mock_result.pose_landmarks = None
        mock_process.return_value = mock_result
        
        result = estimator.process_frame(frame, 0, 0.0)
        
        assert result is None
    
    estimator.close()


@patch('extraction.pose_estimator.cv2.VideoCapture')
@patch('extraction.pose_estimator.cv2.cvtColor')
def test_process_frame_with_pose(mock_cvtcolor, mock_video_cap):
    """포즈 감지 성공"""
    estimator = MediaPipePoseEstimator(enable_hands=False)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = frame
    
    # Mock 포즈 결과
    with patch.object(estimator.pose, 'process') as mock_process:
        mock_result = Mock()
        
        # Mock 랜드마크 (33개)
        mock_landmarks = []
        for i in range(33):
            mock_lm = Mock()
            mock_lm.x = 0.5
            mock_lm.y = 0.5
            mock_lm.z = 0.0
            mock_lm.visibility = 0.9
            mock_landmarks.append(mock_lm)
        
        mock_result.pose_landmarks = Mock()
        mock_result.pose_landmarks.landmark = mock_landmarks
        mock_result.pose_world_landmarks = Mock()
        mock_result.pose_world_landmarks.landmark = mock_landmarks
        
        mock_process.return_value = mock_result
        
        result = estimator.process_frame(frame, 0, 0.0)
        
        assert result is not None
        assert result.frame_idx == 0
        assert result.timestamp == 0.0
        assert len(result.body_landmarks) == 33
        assert result.pose_confidence > 0.0
    
    estimator.close()


@patch('extraction.pose_estimator.cv2.VideoCapture')
def test_process_video_invalid_path(mock_video_cap):
    """잘못된 비디오 경로"""
    mock_cap_instance = Mock()
    mock_cap_instance.isOpened.return_value = False
    mock_video_cap.return_value = mock_cap_instance
    
    estimator = MediaPipePoseEstimator()
    
    with pytest.raises(ValueError, match="Cannot open video"):
        estimator.process_video("invalid.mp4")
    
    estimator.close()


@patch('extraction.pose_estimator.cv2.VideoCapture')
@patch('extraction.pose_estimator.cv2.cvtColor')
def test_process_video_success(mock_cvtcolor, mock_video_cap):
    """비디오 처리 성공"""
    # Mock VideoCapture
    mock_cap_instance = Mock()
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = [30.0, 90]  # FPS, total_frames
    
    # 3프레임 반환 후 종료
    mock_frames = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
    ]
    mock_cap_instance.read.side_effect = [
        (True, mock_frames[0]),
        (True, mock_frames[1]),
        (True, mock_frames[2]),
        (False, None)
    ]
    
    mock_video_cap.return_value = mock_cap_instance
    mock_cvtcolor.side_effect = mock_frames
    
    estimator = MediaPipePoseEstimator(enable_hands=False)
    
    # Mock 포즈 처리 결과
    with patch.object(estimator, 'process_frame') as mock_process_frame:
        mock_process_frame.return_value = FramePose(
            frame_idx=0,
            timestamp=0.0,
            body_landmarks=[PoseLandmark(0, 0, 0, 1) for _ in range(33)],
            pose_confidence=0.9
        )
        
        sequence = estimator.process_video("test.mp4", max_frames=3)
        
        assert isinstance(sequence, VideoPoseSequence)
        assert sequence.video_path == "test.mp4"
        assert sequence.fps == 30.0
        assert len(sequence.frames) > 0
    
    estimator.close()


def test_context_manager():
    """Context manager 사용"""
    with MediaPipePoseEstimator() as estimator:
        assert estimator.pose is not None
    
    # close()가 호출되었는지는 명시적으로 테스트하기 어렵지만
    # 예외가 발생하지 않으면 성공


def test_estimator_close():
    """리소스 해제"""
    estimator = MediaPipePoseEstimator()
    
    # close 전에는 pose가 있음
    assert estimator.pose is not None
    
    estimator.close()
    
    # close 후 - 에러 없이 종료되면 성공
