"""
3D 좌표 변환 및 정규화

기능:
- 2D → 3D 좌표 변환
- 카메라 파라미터 추정
- 깊이 정보 보정
- 좌표계 정규화
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class CameraParameters:
    """카메라 파라미터"""
    # 내부 파라미터 (Intrinsic)
    focal_length: Tuple[float, float]  # (fx, fy)
    principal_point: Tuple[float, float]  # (cx, cy)
    
    # 외부 파라미터 (Extrinsic)
    rotation: Optional[np.ndarray] = None  # 3x3 회전 행렬
    translation: Optional[np.ndarray] = None  # 3x1 이동 벡터
    
    # 왜곡 파라미터
    distortion: Optional[np.ndarray] = None  # (k1, k2, p1, p2, k3)
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """내부 파라미터 행렬 K"""
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K


class CoordinateTransformer:
    """좌표 변환기"""
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        """
        Args:
            image_width: 이미지 너비
            image_height: 이미지 높이
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # 기본 카메라 파라미터 (추정값)
        self.camera = self._estimate_default_camera()
    
    def _estimate_default_camera(self) -> CameraParameters:
        """기본 카메라 파라미터 추정"""
        # 일반적인 스마트폰/웹캠의 파라미터
        # FOV (Field of View) 약 60도 가정
        
        # Focal length 추정
        fov_deg = 60
        fov_rad = np.deg2rad(fov_deg)
        fx = self.image_width / (2 * np.tan(fov_rad / 2))
        fy = fx  # 정사각형 픽셀 가정
        
        # Principal point (이미지 중심)
        cx = self.image_width / 2
        cy = self.image_height / 2
        
        return CameraParameters(
            focal_length=(fx, fy),
            principal_point=(cx, cy),
        )
    
    def normalize_2d_coordinates(
        self,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        2D 좌표 정규화 (0~1 범위 → -1~1 범위)
        
        Args:
            landmarks: [N, 2] 정규화된 좌표 (0~1)
        
        Returns:
            [N, 2] 중심화된 좌표 (-1~1)
        """
        # 0~1 범위를 -1~1 범위로
        normalized = landmarks * 2.0 - 1.0
        
        # 종횡비 보정
        aspect_ratio = self.image_width / self.image_height
        normalized[:, 0] *= aspect_ratio
        
        return normalized
    
    def pixel_to_normalized(
        self,
        pixel_coords: np.ndarray,
    ) -> np.ndarray:
        """
        픽셀 좌표 → 정규화 좌표
        
        Args:
            pixel_coords: [N, 2] 픽셀 좌표
        
        Returns:
            [N, 2] 정규화 좌표 (0~1)
        """
        normalized = pixel_coords.copy().astype(np.float32)
        normalized[:, 0] /= self.image_width
        normalized[:, 1] /= self.image_height
        
        return normalized
    
    def reconstruct_3d_from_depth(
        self,
        landmarks_2d: np.ndarray,
        depth_values: np.ndarray,
    ) -> np.ndarray:
        """
        깊이 정보를 이용한 3D 재구성
        
        Args:
            landmarks_2d: [N, 2] 2D 좌표 (정규화)
            depth_values: [N] 깊이 값
        
        Returns:
            [N, 3] 3D 좌표
        """
        N = landmarks_2d.shape[0]
        
        # 픽셀 좌표로 변환
        pixel_coords = landmarks_2d.copy()
        pixel_coords[:, 0] *= self.image_width
        pixel_coords[:, 1] *= self.image_height
        
        # 카메라 파라미터
        fx, fy = self.camera.focal_length
        cx, cy = self.camera.principal_point
        
        # 3D 좌표 계산
        points_3d = np.zeros((N, 3), dtype=np.float32)
        
        points_3d[:, 2] = depth_values  # Z (깊이)
        points_3d[:, 0] = (pixel_coords[:, 0] - cx) * depth_values / fx  # X
        points_3d[:, 1] = (pixel_coords[:, 1] - cy) * depth_values / fy  # Y
        
        return points_3d
    
    def mediapipe_world_to_camera(
        self,
        world_landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        MediaPipe World 좌표 → 카메라 좌표
        
        MediaPipe의 world landmarks는 골반 중심 기준 좌표계
        
        Args:
            world_landmarks: [N, 3] MediaPipe world 좌표
        
        Returns:
            [N, 3] 카메라 좌표계
        """
        # MediaPipe world coordinates는 이미 미터 단위
        # 좌표계: X=오른쪽, Y=위, Z=카메라 방향
        
        camera_coords = world_landmarks.copy()
        
        # 필요시 좌표계 변환
        # (MediaPipe와 표준 카메라 좌표계가 다를 경우)
        
        return camera_coords
    
    def align_to_reference_frame(
        self,
        landmarks: np.ndarray,
        reference_joint_idx: int = 0,  # 골반 중심
    ) -> np.ndarray:
        """
        기준 관절 중심으로 좌표 정렬
        
        Args:
            landmarks: [N, 3] 3D 좌표
            reference_joint_idx: 기준 관절 인덱스
        
        Returns:
            [N, 3] 정렬된 좌표
        """
        reference_point = landmarks[reference_joint_idx]
        aligned = landmarks - reference_point
        
        return aligned
    
    def compute_relative_coordinates(
        self,
        landmarks: np.ndarray,
        parent_indices: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        상대 좌표 계산 (부모 관절 기준)
        
        Args:
            landmarks: [N, 3] 절대 좌표
            parent_indices: {child_idx: parent_idx} 부모-자식 관계
        
        Returns:
            [N, 3] 상대 좌표
        """
        if parent_indices is None:
            # MediaPipe Pose의 기본 계층 구조
            parent_indices = self._get_mediapipe_hierarchy()
        
        relative = landmarks.copy()
        
        for child_idx, parent_idx in parent_indices.items():
            if parent_idx >= 0 and parent_idx < len(landmarks):
                relative[child_idx] = landmarks[child_idx] - landmarks[parent_idx]
        
        return relative
    
    @staticmethod
    def _get_mediapipe_hierarchy() -> Dict[int, int]:
        """MediaPipe Pose 계층 구조"""
        # 간략화된 버전 (주요 관절만)
        return {
            # 왼쪽 팔
            11: 0,  # 왼쪽 어깨 → 골반
            13: 11,  # 왼쪽 팔꿈치 → 어깨
            15: 13,  # 왼쪽 손목 → 팔꿈치
            
            # 오른쪽 팔
            12: 0,  # 오른쪽 어깨 → 골반
            14: 12,  # 오른쪽 팔꿈치 → 어깨
            16: 14,  # 오른쪽 손목 → 팔꿈치
            
            # 왼쪽 다리
            23: 0,  # 왼쪽 엉덩이 → 골반
            25: 23,  # 왼쪽 무릎 → 엉덩이
            27: 25,  # 왼쪽 발목 → 무릎
            
            # 오른쪽 다리
            24: 0,  # 오른쪽 엉덩이 → 골반
            26: 24,  # 오른쪽 무릎 → 엉덩이
            28: 26,  # 오른쪽 발목 → 무릎
        }
    
    def scale_normalization(
        self,
        landmarks: np.ndarray,
        target_height: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        신체 크기 정규화
        
        Args:
            landmarks: [N, 3] 3D 좌표
            target_height: 목표 신장 (미터)
        
        Returns:
            (정규화된 좌표, 스케일 팩터)
        """
        # 신장 추정 (머리 ~ 발목)
        # MediaPipe: 0=nose, 27=left_ankle, 28=right_ankle
        
        nose_idx = 0
        left_ankle_idx = 27
        right_ankle_idx = 28
        
        head_pos = landmarks[nose_idx]
        left_foot = landmarks[left_ankle_idx]
        right_foot = landmarks[right_ankle_idx]
        foot_pos = (left_foot + right_foot) / 2
        
        current_height = np.linalg.norm(head_pos - foot_pos)
        
        if current_height < 1e-6:
            return landmarks, 1.0
        
        scale_factor = target_height / current_height
        normalized = landmarks * scale_factor
        
        return normalized, scale_factor
    
    def rotation_normalization(
        self,
        landmarks: np.ndarray,
    ) -> Tuple[np.ndarray, Rotation]:
        """
        회전 정규화 (정면 바라보도록)
        
        Args:
            landmarks: [N, 3] 3D 좌표
        
        Returns:
            (정규화된 좌표, 회전 객체)
        """
        # 어깨 벡터로 방향 추정
        left_shoulder_idx = 11
        right_shoulder_idx = 12
        
        left_shoulder = landmarks[left_shoulder_idx]
        right_shoulder = landmarks[right_shoulder_idx]
        
        # 어깨 벡터 (왼쪽 → 오른쪽)
        shoulder_vec = right_shoulder - left_shoulder
        
        # Y축 정렬
        target_vec = np.array([1, 0, 0])  # X축 방향
        
        # 회전 계산
        rotation = self._rotation_from_vectors(shoulder_vec, target_vec)
        
        # 회전 적용
        rotated = rotation.apply(landmarks)
        
        return rotated, rotation
    
    @staticmethod
    def _rotation_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> Rotation:
        """두 벡터 사이의 회전 계산"""
        # 벡터 정규화
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        
        # 회전축
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # 평행한 경우
            return Rotation.identity()
        
        axis = axis / axis_norm
        
        # 회전 각도
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        # Rotation 객체 생성
        rotation = Rotation.from_rotvec(axis * angle)
        
        return rotation
