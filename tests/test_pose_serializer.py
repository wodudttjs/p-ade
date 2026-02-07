"""
포즈 직렬화 테스트
"""

import pytest
import json
import numpy as np
from pathlib import Path

from extraction.pose_estimator import VideoPoseSequence, FramePose, PoseLandmark
from extraction.pose_serializer import PoseSerializer


@pytest.fixture
def sample_sequence():
    """샘플 포즈 시퀀스"""
    sequence = VideoPoseSequence(
        video_path="/path/to/video.mp4",
        fps=30.0,
        total_frames=900
    )
    
    # 몇 개의 프레임 추가
    for i in range(5):
        body_landmarks = [
            PoseLandmark(x=float(j) * 0.01, y=float(j) * 0.02, z=float(j) * 0.03, visibility=0.9)
            for j in range(33)
        ]
        
        hand_landmarks = [
            PoseLandmark(x=float(j) * 0.01, y=float(j) * 0.02, z=0.0, visibility=1.0)
            for j in range(21)
        ]
        
        frame_pose = FramePose(
            frame_idx=i,
            timestamp=i / 30.0,
            body_landmarks=body_landmarks,
            body_world_landmarks=body_landmarks,  # 실제로는 다를 수 있음
            left_hand_landmarks=hand_landmarks,
            right_hand_landmarks=hand_landmarks,
            pose_confidence=0.9
        )
        
        sequence.frames.append(frame_pose)
    
    return sequence


def test_save_load_json(sample_sequence, tmp_path):
    """JSON 저장/로드"""
    output_path = tmp_path / "pose.json"
    
    # 저장
    saved_path = PoseSerializer.save_json(sample_sequence, output_path)
    assert saved_path.exists()
    
    # JSON 내용 확인
    with open(saved_path, 'r') as f:
        data = json.load(f)
    
    assert 'metadata' in data
    assert 'frames' in data
    assert data['metadata']['fps'] == 30.0
    assert len(data['frames']) == 5
    
    # 로드
    loaded_sequence = PoseSerializer.load_json(saved_path)
    
    assert loaded_sequence.video_path == sample_sequence.video_path
    assert loaded_sequence.fps == sample_sequence.fps
    assert len(loaded_sequence.frames) == len(sample_sequence.frames)
    
    # 첫 번째 프레임 비교
    original_frame = sample_sequence.frames[0]
    loaded_frame = loaded_sequence.frames[0]
    
    assert loaded_frame.frame_idx == original_frame.frame_idx
    assert loaded_frame.timestamp == original_frame.timestamp
    assert len(loaded_frame.body_landmarks) == len(original_frame.body_landmarks)


def test_save_load_numpy(sample_sequence, tmp_path):
    """NumPy 저장/로드"""
    output_path = tmp_path / "pose.npz"
    
    # 저장
    saved_path = PoseSerializer.save_numpy(sample_sequence, output_path)
    assert saved_path.exists()
    
    # 로드
    loaded_data = PoseSerializer.load_numpy(saved_path)
    
    assert 'body' in loaded_data
    assert 'body_world' in loaded_data
    assert 'timestamps' in loaded_data
    assert 'confidence' in loaded_data
    assert 'metadata' in loaded_data
    
    # 데이터 형상 확인
    assert loaded_data['body'].shape == (5, 33, 3)
    assert loaded_data['left_hand'].shape == (5, 21, 3)
    assert loaded_data['timestamps'].shape == (5,)


def test_save_load_hdf5(sample_sequence, tmp_path):
    """HDF5 저장/로드"""
    output_path = tmp_path / "pose.h5"
    
    # 저장
    saved_path = PoseSerializer.save_hdf5(sample_sequence, output_path)
    assert saved_path.exists()
    
    # 로드
    loaded_data = PoseSerializer.load_hdf5(saved_path)
    
    assert 'metadata' in loaded_data
    assert loaded_data['metadata']['fps'] == 30.0
    assert loaded_data['metadata']['num_pose_frames'] == 5
    
    assert 'body' in loaded_data
    assert 'timestamps' in loaded_data
    
    # 데이터 형상 확인
    assert loaded_data['body'].shape == (5, 33, 3)


def test_save_load_pickle(sample_sequence, tmp_path):
    """Pickle 저장/로드"""
    output_path = tmp_path / "pose.pkl"
    
    # 저장
    saved_path = PoseSerializer.save_pickle(sample_sequence, output_path)
    assert saved_path.exists()
    
    # 로드
    loaded_sequence = PoseSerializer.load_pickle(saved_path)
    
    assert isinstance(loaded_sequence, VideoPoseSequence)
    assert loaded_sequence.video_path == sample_sequence.video_path
    assert loaded_sequence.fps == sample_sequence.fps
    assert len(loaded_sequence.frames) == len(sample_sequence.frames)


def test_get_format_from_extension():
    """확장자에서 포맷 추론"""
    assert PoseSerializer.get_format_from_extension(Path("test.json")) == "json"
    assert PoseSerializer.get_format_from_extension(Path("test.npz")) == "numpy"
    assert PoseSerializer.get_format_from_extension(Path("test.h5")) == "hdf5"
    assert PoseSerializer.get_format_from_extension(Path("test.hdf5")) == "hdf5"
    assert PoseSerializer.get_format_from_extension(Path("test.pkl")) == "pickle"
    assert PoseSerializer.get_format_from_extension(Path("test.pickle")) == "pickle"
    assert PoseSerializer.get_format_from_extension(Path("test.txt")) is None


def test_save_auto_format_json(sample_sequence, tmp_path):
    """자동 포맷 감지 - JSON"""
    output_path = tmp_path / "pose.json"
    
    saved_path = PoseSerializer.save(sample_sequence, output_path)
    
    assert saved_path.exists()
    
    # JSON으로 로드 가능한지 확인
    with open(saved_path, 'r') as f:
        data = json.load(f)
    assert 'metadata' in data


def test_save_auto_format_numpy(sample_sequence, tmp_path):
    """자동 포맷 감지 - NumPy"""
    output_path = tmp_path / "pose.npz"
    
    saved_path = PoseSerializer.save(sample_sequence, output_path)
    
    assert saved_path.exists()
    
    # NumPy로 로드 가능한지 확인
    data = np.load(saved_path)
    assert 'body' in data


def test_save_explicit_format(sample_sequence, tmp_path):
    """명시적 포맷 지정"""
    output_path = tmp_path / "pose_data"  # 확장자 없음
    
    saved_path = PoseSerializer.save(sample_sequence, output_path, format='json')
    
    assert saved_path.exists()


def test_save_unsupported_format(sample_sequence, tmp_path):
    """지원하지 않는 포맷"""
    output_path = tmp_path / "pose.txt"
    
    with pytest.raises(ValueError, match="Unsupported format"):
        PoseSerializer.save(sample_sequence, output_path)


def test_get_file_size_mb(sample_sequence, tmp_path):
    """파일 크기 확인"""
    output_path = tmp_path / "pose.json"
    
    PoseSerializer.save_json(sample_sequence, output_path)
    
    size_mb = PoseSerializer.get_file_size_mb(output_path)
    
    assert size_mb > 0
    assert isinstance(size_mb, float)


def test_json_with_no_hands(tmp_path):
    """손 데이터 없는 시퀀스"""
    sequence = VideoPoseSequence(
        video_path="test.mp4",
        fps=30.0,
        total_frames=90
    )
    
    # 손 데이터 없이 body만
    body_landmarks = [
        PoseLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
        for _ in range(33)
    ]
    
    frame_pose = FramePose(
        frame_idx=0,
        timestamp=0.0,
        body_landmarks=body_landmarks,
        body_world_landmarks=body_landmarks,
        left_hand_landmarks=None,
        right_hand_landmarks=None,
        pose_confidence=0.9
    )
    
    sequence.frames.append(frame_pose)
    
    output_path = tmp_path / "pose_no_hands.json"
    
    # 저장/로드
    PoseSerializer.save_json(sequence, output_path)
    loaded = PoseSerializer.load_json(output_path)
    
    assert len(loaded.frames) == 1
    assert loaded.frames[0].left_hand_landmarks is None
    assert loaded.frames[0].right_hand_landmarks is None


def test_empty_sequence(tmp_path):
    """빈 시퀀스"""
    sequence = VideoPoseSequence(
        video_path="empty.mp4",
        fps=30.0,
        total_frames=0
    )
    
    output_path = tmp_path / "empty.json"
    
    PoseSerializer.save_json(sequence, output_path)
    loaded = PoseSerializer.load_json(output_path)
    
    assert len(loaded.frames) == 0
    assert loaded.video_path == "empty.mp4"


def test_numpy_data_integrity(sample_sequence, tmp_path):
    """NumPy 데이터 무결성"""
    output_path = tmp_path / "pose.npz"
    
    # 원본 NumPy 데이터
    original_numpy = sample_sequence.get_as_numpy()
    
    # 저장/로드
    PoseSerializer.save_numpy(sample_sequence, output_path)
    loaded_numpy = PoseSerializer.load_numpy(output_path)
    
    # 데이터 비교
    assert np.allclose(original_numpy['body'], loaded_numpy['body'])
    assert np.allclose(original_numpy['timestamps'], loaded_numpy['timestamps'])
    assert np.allclose(original_numpy['confidence'], loaded_numpy['confidence'])


def test_hdf5_compression(sample_sequence, tmp_path):
    """HDF5 압축 효과"""
    output_path = tmp_path / "pose.h5"
    
    saved_path = PoseSerializer.save_hdf5(sample_sequence, output_path)
    
    # 파일이 생성되었고 크기가 있는지만 확인
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_multiple_formats_same_data(sample_sequence, tmp_path):
    """여러 포맷으로 저장 후 데이터 일관성"""
    json_path = tmp_path / "pose.json"
    npz_path = tmp_path / "pose.npz"
    
    # JSON 저장/로드
    PoseSerializer.save_json(sample_sequence, json_path)
    json_loaded = PoseSerializer.load_json(json_path)
    
    # NumPy 저장/로드
    PoseSerializer.save_numpy(sample_sequence, npz_path)
    npz_loaded = PoseSerializer.load_numpy(npz_path)
    
    # NumPy 변환 후 비교
    json_numpy = json_loaded.get_as_numpy()
    
    assert np.allclose(json_numpy['body'], npz_loaded['body'])
    assert np.allclose(json_numpy['timestamps'], npz_loaded['timestamps'])
