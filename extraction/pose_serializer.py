"""
포즈 데이터 직렬화

포즈 시퀀스를 다양한 형식으로 저장하고 로드합니다.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import h5py

from extraction.pose_estimator import VideoPoseSequence, FramePose, PoseLandmark
from core.logging_config import logger


class PoseSerializer:
    """포즈 데이터 직렬화/역직렬화"""
    
    @staticmethod
    def save_json(sequence: VideoPoseSequence, output_path: Path) -> Path:
        """
        JSON 형식으로 저장
        
        Args:
            sequence: 포즈 시퀀스
            output_path: 출력 경로
            
        Returns:
            저장된 파일 경로
        """
        data = {
            'metadata': {
                'video_path': sequence.video_path,
                'fps': sequence.fps,
                'total_frames': sequence.total_frames,
                'num_pose_frames': len(sequence.frames),
            },
            'frames': []
        }
        
        for frame_pose in sequence.frames:
            frame_data = {
                'frame_idx': frame_pose.frame_idx,
                'timestamp': frame_pose.timestamp,
                'pose_confidence': frame_pose.pose_confidence,
                'body_landmarks': [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in frame_pose.body_landmarks
                ],
            }
            
            if frame_pose.body_world_landmarks:
                frame_data['body_world_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in frame_pose.body_world_landmarks
                ]
            
            if frame_pose.left_hand_landmarks:
                frame_data['left_hand_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in frame_pose.left_hand_landmarks
                ]
            
            if frame_pose.right_hand_landmarks:
                frame_data['right_hand_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in frame_pose.right_hand_landmarks
                ]
            
            data['frames'].append(frame_data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved pose data to JSON: {output_path}")
        return output_path
    
    @staticmethod
    def load_json(input_path: Path) -> VideoPoseSequence:
        """
        JSON에서 로드
        
        Args:
            input_path: 입력 경로
            
        Returns:
            VideoPoseSequence
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        sequence = VideoPoseSequence(
            video_path=metadata['video_path'],
            fps=metadata['fps'],
            total_frames=metadata['total_frames'],
        )
        
        for frame_data in data['frames']:
            body_landmarks = [
                PoseLandmark(**lm_data)
                for lm_data in frame_data['body_landmarks']
            ]
            
            body_world_landmarks = None
            if 'body_world_landmarks' in frame_data:
                body_world_landmarks = [
                    PoseLandmark(**lm_data)
                    for lm_data in frame_data['body_world_landmarks']
                ]
            
            left_hand = None
            if 'left_hand_landmarks' in frame_data:
                left_hand = [
                    PoseLandmark(**lm_data)
                    for lm_data in frame_data['left_hand_landmarks']
                ]
            
            right_hand = None
            if 'right_hand_landmarks' in frame_data:
                right_hand = [
                    PoseLandmark(**lm_data)
                    for lm_data in frame_data['right_hand_landmarks']
                ]
            
            frame_pose = FramePose(
                frame_idx=frame_data['frame_idx'],
                timestamp=frame_data['timestamp'],
                body_landmarks=body_landmarks,
                body_world_landmarks=body_world_landmarks or [],
                left_hand_landmarks=left_hand,
                right_hand_landmarks=right_hand,
                pose_confidence=frame_data['pose_confidence'],
            )
            
            sequence.frames.append(frame_pose)
        
        logger.info(f"Loaded pose data from JSON: {input_path}")
        return sequence
    
    @staticmethod
    def save_numpy(sequence: VideoPoseSequence, output_path: Path) -> Path:
        """
        NumPy NPZ 형식으로 저장
        
        Args:
            sequence: 포즈 시퀀스
            output_path: 출력 경로
            
        Returns:
            저장된 파일 경로
        """
        numpy_data = sequence.get_as_numpy()
        
        # 메타데이터 추가
        numpy_data['metadata'] = np.array([
            sequence.fps,
            sequence.total_frames,
            len(sequence.frames)
        ])
        numpy_data['video_path'] = np.array(sequence.video_path)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path, **numpy_data)
        
        logger.info(f"Saved pose data to NumPy: {output_path}")
        return output_path
    
    @staticmethod
    def load_numpy(input_path: Path) -> Dict[str, np.ndarray]:
        """
        NumPy NPZ에서 로드
        
        Args:
            input_path: 입력 경로
            
        Returns:
            NumPy 데이터 딕셔너리
        """
        data = np.load(input_path, allow_pickle=True)
        
        result = {}
        for key in data.keys():
            result[key] = data[key]
        
        logger.info(f"Loaded pose data from NumPy: {input_path}")
        return result
    
    @staticmethod
    def save_hdf5(sequence: VideoPoseSequence, output_path: Path) -> Path:
        """
        HDF5 형식으로 저장
        
        Args:
            sequence: 포즈 시퀀스
            output_path: 출력 경로
            
        Returns:
            저장된 파일 경로
        """
        numpy_data = sequence.get_as_numpy()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # 메타데이터
            f.attrs['video_path'] = sequence.video_path
            f.attrs['fps'] = sequence.fps
            f.attrs['total_frames'] = sequence.total_frames
            f.attrs['num_pose_frames'] = len(sequence.frames)
            
            # 데이터셋
            for key, value in numpy_data.items():
                f.create_dataset(key, data=value, compression='gzip')
        
        logger.info(f"Saved pose data to HDF5: {output_path}")
        return output_path
    
    @staticmethod
    def load_hdf5(input_path: Path) -> Dict[str, Any]:
        """
        HDF5에서 로드
        
        Args:
            input_path: 입력 경로
            
        Returns:
            데이터 딕셔너리 (메타데이터 + NumPy 배열)
        """
        with h5py.File(input_path, 'r') as f:
            # 메타데이터
            result = {
                'metadata': {
                    'video_path': f.attrs['video_path'],
                    'fps': f.attrs['fps'],
                    'total_frames': f.attrs['total_frames'],
                    'num_pose_frames': f.attrs['num_pose_frames'],
                }
            }
            
            # 데이터셋
            for key in f.keys():
                result[key] = f[key][:]
        
        logger.info(f"Loaded pose data from HDF5: {input_path}")
        return result
    
    @staticmethod
    def save_pickle(sequence: VideoPoseSequence, output_path: Path) -> Path:
        """
        Pickle 형식으로 저장 (Python 객체 그대로)
        
        Args:
            sequence: 포즈 시퀀스
            output_path: 출력 경로
            
        Returns:
            저장된 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(sequence, f)
        
        logger.info(f"Saved pose data to Pickle: {output_path}")
        return output_path
    
    @staticmethod
    def load_pickle(input_path: Path) -> VideoPoseSequence:
        """
        Pickle에서 로드
        
        Args:
            input_path: 입력 경로
            
        Returns:
            VideoPoseSequence
        """
        with open(input_path, 'rb') as f:
            sequence = pickle.load(f)
        
        logger.info(f"Loaded pose data from Pickle: {input_path}")
        return sequence
    
    @staticmethod
    def get_format_from_extension(path: Path) -> Optional[str]:
        """
        파일 확장자에서 포맷 추론
        
        Args:
            path: 파일 경로
            
        Returns:
            포맷 이름 (json, npz, h5, pkl)
        """
        suffix = path.suffix.lower()
        
        format_map = {
            '.json': 'json',
            '.npz': 'numpy',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
        }
        
        return format_map.get(suffix)
    
    @staticmethod
    def save(
        sequence: VideoPoseSequence,
        output_path: Path,
        format: Optional[str] = None
    ) -> Path:
        """
        자동 포맷 감지 저장
        
        Args:
            sequence: 포즈 시퀀스
            output_path: 출력 경로
            format: 포맷 (None이면 확장자로 추론)
            
        Returns:
            저장된 파일 경로
        """
        if format is None:
            format = PoseSerializer.get_format_from_extension(output_path)
        
        if format == 'json':
            return PoseSerializer.save_json(sequence, output_path)
        elif format == 'numpy':
            return PoseSerializer.save_numpy(sequence, output_path)
        elif format == 'hdf5':
            return PoseSerializer.save_hdf5(sequence, output_path)
        elif format == 'pickle':
            return PoseSerializer.save_pickle(sequence, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def get_file_size_mb(path: Path) -> float:
        """파일 크기 (MB)"""
        return path.stat().st_size / 1024 / 1024
