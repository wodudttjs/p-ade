"""
FR-4.4: Data Augmentation

데이터 증강으로 다양성 확보
- Spatial Augmentation (미러링, 스케일, 노이즈)
- Temporal Augmentation (속도, 오프셋, 부분 추출)
- Viewpoint Simulation (카메라 각도, 가상 시점)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import uuid

from core.logging_config import setup_logger
from transformation.spec import (
    TransformConfig,
    AugmentConfig,
    SpatialAugmentConfig,
    TemporalAugmentConfig,
    ViewpointAugmentConfig,
    MEDIAPIPE_LANDMARKS,
    LEFT_RIGHT_PAIRS,
    get_mirror_swap_indices,
)

logger = setup_logger(__name__)


@dataclass
class AugmentedData:
    """증강된 데이터"""
    pose: np.ndarray  # [T, J, 3]
    timestamps: np.ndarray  # [T]
    actions: Optional[np.ndarray] = None  # [T-1, A]
    aug_id: str = ""
    aug_type: str = ""
    aug_params: Dict[str, Any] = None
    source_episode_id: str = ""
    
    def __post_init__(self):
        if self.aug_params is None:
            self.aug_params = {}
        if not self.aug_id:
            self.aug_id = f"aug_{uuid.uuid4().hex[:8]}"


class SpatialAugmenter:
    """
    Task 4.4.1: Spatial Augmentation
    
    미러링, 스케일 변화, 노이즈 추가
    """
    
    def __init__(self, config: Optional[SpatialAugmentConfig] = None):
        self.config = config or SpatialAugmentConfig()
        
    def mirror(
        self,
        pose: np.ndarray,
        joint_names: Optional[List[str]] = None,
    ) -> AugmentedData:
        """
        좌우 미러링
        
        Args:
            pose: [T, J, 3] 포즈
            joint_names: 관절 이름 리스트 (좌우 스왑용)
            
        Returns:
            AugmentedData: 미러링된 데이터
        """
        mirrored = pose.copy()
        
        # X축 반전 (바디 프레임에서 좌우 축)
        mirrored[:, :, 0] = -mirrored[:, :, 0]
        
        # 좌우 관절 스왑
        if joint_names is None:
            # 전체 MediaPipe 관절 사용
            joint_names = list(MEDIAPIPE_LANDMARKS.keys())
            
        swap_map = get_mirror_swap_indices(joint_names)
        
        if swap_map:
            for left_idx, right_idx in swap_map.items():
                if left_idx < right_idx:  # 한 번만 스왑
                    temp = mirrored[:, left_idx].copy()
                    mirrored[:, left_idx] = mirrored[:, right_idx]
                    mirrored[:, right_idx] = temp
                    
        return AugmentedData(
            pose=mirrored,
            timestamps=np.arange(len(mirrored)) / 30.0,  # 기본 타임스탬프
            aug_type="mirror",
            aug_params={"axis": "x"},
        )
    
    def scale(
        self,
        pose: np.ndarray,
        scale_range: Optional[Tuple[float, float]] = None,
    ) -> AugmentedData:
        """
        스케일 변화
        
        Args:
            pose: [T, J, 3] 포즈
            scale_range: (min, max) 스케일 범위
            
        Returns:
            AugmentedData: 스케일 변환된 데이터
        """
        if scale_range is None:
            scale_range = self.config.scale_range
            
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        scaled = pose * scale_factor
        
        return AugmentedData(
            pose=scaled,
            timestamps=np.arange(len(scaled)) / 30.0,
            aug_type="scale",
            aug_params={"scale_factor": scale_factor},
        )
    
    def add_noise(
        self,
        pose: np.ndarray,
        sigma: Optional[float] = None,
        conf: Optional[np.ndarray] = None,
    ) -> AugmentedData:
        """
        Gaussian 노이즈 추가
        
        Args:
            pose: [T, J, 3] 포즈
            sigma: 노이즈 표준편차
            conf: [T, J] 신뢰도 (저신뢰 관절에는 노이즈 가중)
            
        Returns:
            AugmentedData: 노이즈 추가된 데이터
        """
        if sigma is None:
            sigma = self.config.noise_sigma
            
        noise = np.random.normal(0, sigma, pose.shape)
        
        # 저신뢰도 관절에는 노이즈 줄이기 (선택적)
        if conf is not None:
            # conf가 낮으면 노이즈도 줄임 (이미 불확실하므로)
            weight = conf[:, :, np.newaxis]  # [T, J, 1]
            noise = noise * weight
            
        noisy = pose + noise
        
        return AugmentedData(
            pose=noisy,
            timestamps=np.arange(len(noisy)) / 30.0,
            aug_type="noise",
            aug_params={"sigma": sigma},
        )
    
    def augment(
        self,
        pose: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        conf: Optional[np.ndarray] = None,
    ) -> List[AugmentedData]:
        """
        모든 공간 증강 적용
        
        Returns:
            증강된 데이터 리스트
        """
        augmented = []
        
        if timestamps is None:
            timestamps = np.arange(len(pose)) / 30.0
            
        if self.config.enable_mirror:
            result = self.mirror(pose, joint_names)
            result.timestamps = timestamps.copy()
            augmented.append(result)
            
        if self.config.enable_scale:
            result = self.scale(pose)
            result.timestamps = timestamps.copy()
            augmented.append(result)
            
        if self.config.enable_noise:
            result = self.add_noise(pose, conf=conf)
            result.timestamps = timestamps.copy()
            augmented.append(result)
            
        return augmented


class TemporalAugmenter:
    """
    Task 4.4.2: Temporal Augmentation
    
    속도 변화, 시작점 오프셋, 부분 구간 추출
    """
    
    def __init__(self, config: Optional[TemporalAugmentConfig] = None):
        self.config = config or TemporalAugmentConfig()
        
    def speed_change(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        speed_range: Optional[Tuple[float, float]] = None,
    ) -> AugmentedData:
        """
        속도 변화 (Time warping)
        
        Args:
            pose: [T, J, 3] 포즈
            timestamps: [T] 타임스탬프
            speed_range: (min, max) 속도 배율
            
        Returns:
            AugmentedData: 속도 변환된 데이터
        """
        if speed_range is None:
            speed_range = self.config.speed_range
            
        speed_factor = np.random.uniform(speed_range[0], speed_range[1])
        
        T = len(timestamps)
        original_duration = timestamps[-1] - timestamps[0]
        new_duration = original_duration / speed_factor
        
        # 새 타임스탬프 생성
        new_timestamps = np.linspace(
            timestamps[0],
            timestamps[0] + new_duration,
            int(T / speed_factor),
        )
        
        # 보간
        T_new = len(new_timestamps)
        new_pose = np.zeros((T_new, pose.shape[1], pose.shape[2]))
        
        for j in range(pose.shape[1]):
            for c in range(pose.shape[2]):
                interp_func = interp1d(
                    timestamps,
                    pose[:, j, c],
                    kind="linear",
                    fill_value="extrapolate",
                )
                new_pose[:, j, c] = interp_func(new_timestamps)
                
        return AugmentedData(
            pose=new_pose,
            timestamps=new_timestamps,
            aug_type="speed_change",
            aug_params={"speed_factor": speed_factor},
        )
    
    def random_offset(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        max_offset_sec: Optional[float] = None,
    ) -> AugmentedData:
        """
        시작점 랜덤 오프셋
        
        Args:
            pose: [T, J, 3] 포즈
            timestamps: [T] 타임스탬프
            max_offset_sec: 최대 오프셋 (초)
            
        Returns:
            AugmentedData: 오프셋 적용된 데이터
        """
        if max_offset_sec is None:
            max_offset_sec = self.config.max_offset_sec
            
        duration = timestamps[-1] - timestamps[0]
        max_offset_frames = int(max_offset_sec / (timestamps[1] - timestamps[0]))
        max_offset_frames = min(max_offset_frames, len(pose) // 4)  # 최대 25% 오프셋
        
        if max_offset_frames <= 0:
            return AugmentedData(
                pose=pose.copy(),
                timestamps=timestamps.copy(),
                aug_type="offset",
                aug_params={"offset_frames": 0},
            )
            
        offset = np.random.randint(0, max_offset_frames + 1)
        
        new_pose = pose[offset:]
        new_timestamps = timestamps[offset:] - timestamps[offset]  # 0부터 시작
        
        return AugmentedData(
            pose=new_pose,
            timestamps=new_timestamps,
            aug_type="offset",
            aug_params={"offset_frames": offset, "offset_sec": timestamps[offset] - timestamps[0]},
        )
    
    def random_crop(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        min_duration_sec: Optional[float] = None,
    ) -> AugmentedData:
        """
        랜덤 구간 추출
        
        Args:
            pose: [T, J, 3] 포즈
            timestamps: [T] 타임스탬프
            min_duration_sec: 최소 길이 (초)
            
        Returns:
            AugmentedData: 추출된 구간
        """
        if min_duration_sec is None:
            min_duration_sec = self.config.min_duration_sec
            
        T = len(pose)
        dt = timestamps[1] - timestamps[0] if T > 1 else 0.0333
        min_frames = int(min_duration_sec / dt)
        
        if min_frames >= T:
            return AugmentedData(
                pose=pose.copy(),
                timestamps=timestamps.copy(),
                aug_type="crop",
                aug_params={"start": 0, "end": T},
            )
            
        # 랜덤 시작점과 길이
        max_start = T - min_frames
        start = np.random.randint(0, max_start + 1)
        
        # 랜덤 길이 (min_frames ~ T - start)
        max_length = T - start
        length = np.random.randint(min_frames, max_length + 1)
        end = start + length
        
        new_pose = pose[start:end]
        new_timestamps = timestamps[start:end] - timestamps[start]
        
        return AugmentedData(
            pose=new_pose,
            timestamps=new_timestamps,
            aug_type="crop",
            aug_params={"start": start, "end": end, "length": length},
        )
    
    def augment(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        actions: Optional[np.ndarray] = None,
    ) -> List[AugmentedData]:
        """
        모든 시간 증강 적용
        
        Returns:
            증강된 데이터 리스트
        """
        augmented = []
        
        if self.config.enable_speed:
            result = self.speed_change(pose, timestamps)
            augmented.append(result)
            
        if self.config.enable_offset:
            result = self.random_offset(pose, timestamps)
            augmented.append(result)
            
        if self.config.enable_crop:
            result = self.random_crop(pose, timestamps)
            augmented.append(result)
            
        return augmented


class ViewpointAugmenter:
    """
    Task 4.4.3: Viewpoint Simulation
    
    카메라 각도 변환, 가상 시점 생성, Depth 보정
    """
    
    def __init__(self, config: Optional[ViewpointAugmentConfig] = None):
        self.config = config or ViewpointAugmentConfig()
        self.eps = 1e-8
        
    def rotate_viewpoint(
        self,
        pose: np.ndarray,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> AugmentedData:
        """
        가상 카메라 회전 적용
        
        Args:
            pose: [T, J, 3] 3D 포즈
            yaw: Yaw 각도 (도)
            pitch: Pitch 각도 (도)
            roll: Roll 각도 (도)
            
        Returns:
            AugmentedData: 회전된 데이터
        """
        # 랜덤 각도 생성
        if yaw is None:
            yaw = np.random.uniform(*self.config.yaw_range)
        if pitch is None:
            pitch = np.random.uniform(*self.config.pitch_range)
        if roll is None:
            roll = np.random.uniform(*self.config.roll_range)
            
        # 회전 행렬 생성 (Euler angles)
        rotation = Rotation.from_euler("yxz", [yaw, pitch, roll], degrees=True)
        rot_matrix = rotation.as_matrix()
        
        # 전체 프레임에 회전 적용
        T, J, _ = pose.shape
        rotated = np.zeros_like(pose)
        
        for t in range(T):
            rotated[t] = pose[t] @ rot_matrix.T
            
        return AugmentedData(
            pose=rotated,
            timestamps=np.arange(T) / 30.0,
            aug_type="viewpoint_rotation",
            aug_params={"yaw": yaw, "pitch": pitch, "roll": roll},
        )
    
    def generate_multi_view(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        n_views: int = 3,
    ) -> List[AugmentedData]:
        """
        다중 가상 시점 생성
        
        Args:
            pose: [T, J, 3] 포즈
            timestamps: [T] 타임스탬프
            n_views: 생성할 시점 수
            
        Returns:
            다중 시점 데이터 리스트
        """
        views = []
        
        for i in range(n_views):
            result = self.rotate_viewpoint(pose)
            result.timestamps = timestamps.copy()
            result.aug_id = f"view_{i:02d}_{result.aug_id}"
            views.append(result)
            
        return views
    
    def normalize_depth(
        self,
        pose: np.ndarray,
        method: str = "median",
    ) -> AugmentedData:
        """
        Depth (Z축) 정규화
        
        Args:
            pose: [T, J, 3] 포즈
            method: "median" (중앙값 기준) 또는 "range" (범위 정규화)
            
        Returns:
            AugmentedData: Depth 정규화된 데이터
        """
        normalized = pose.copy()
        z = pose[:, :, 2]  # [T, J]
        
        if method == "median":
            # 중앙값 기준 정규화
            median_z = np.median(z)
            mad_z = np.median(np.abs(z - median_z))
            normalized[:, :, 2] = (z - median_z) / (mad_z + self.eps)
            
        elif method == "range":
            # 범위 정규화 [-1, 1]
            z_min, z_max = z.min(), z.max()
            z_range = z_max - z_min
            if z_range > self.eps:
                normalized[:, :, 2] = 2 * (z - z_min) / z_range - 1
                
        elif method == "clip":
            # Clipping
            z_median = np.median(z)
            z_std = np.std(z)
            z_min = z_median - 3 * z_std
            z_max = z_median + 3 * z_std
            normalized[:, :, 2] = np.clip(z, z_min, z_max)
            
        return AugmentedData(
            pose=normalized,
            timestamps=np.arange(len(pose)) / 30.0,
            aug_type="depth_normalize",
            aug_params={"method": method},
        )
    
    def augment(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[AugmentedData]:
        """
        모든 시점 증강 적용
        """
        augmented = []
        
        if self.config.enable_rotation:
            result = self.rotate_viewpoint(pose)
            result.timestamps = timestamps.copy()
            augmented.append(result)
            
        if self.config.enable_depth_normalize:
            result = self.normalize_depth(pose)
            result.timestamps = timestamps.copy()
            augmented.append(result)
            
        return augmented


class DataAugmentationPipeline:
    """
    데이터 증강 통합 파이프라인
    """
    
    def __init__(self, config: Optional[AugmentConfig] = None):
        self.config = config or AugmentConfig()
        
        self.spatial = SpatialAugmenter(self.config.spatial)
        self.temporal = TemporalAugmenter(self.config.temporal)
        self.viewpoint = ViewpointAugmenter(self.config.viewpoint)
        
    def augment(
        self,
        pose: np.ndarray,
        timestamps: np.ndarray,
        actions: Optional[np.ndarray] = None,
        source_episode_id: str = "",
        joint_names: Optional[List[str]] = None,
        conf: Optional[np.ndarray] = None,
        augment_types: Optional[List[str]] = None,
    ) -> List[AugmentedData]:
        """
        전체 증강 파이프라인 실행
        
        Args:
            pose: [T, J, 3] 포즈
            timestamps: [T] 타임스탬프
            actions: [T-1, A] 액션
            source_episode_id: 원본 에피소드 ID
            joint_names: 관절 이름 (미러링용)
            conf: [T, J] 신뢰도
            augment_types: 적용할 증강 타입 (None이면 전체)
            
        Returns:
            증강된 데이터 리스트
        """
        if not self.config.enable:
            return []
            
        augmented = []
        
        # 적용할 증강 타입 결정
        if augment_types is None:
            augment_types = ["spatial", "temporal", "viewpoint"]
            
        # Spatial augmentation
        if "spatial" in augment_types:
            spatial_results = self.spatial.augment(
                pose, timestamps, joint_names, conf
            )
            for result in spatial_results:
                result.source_episode_id = source_episode_id
                augmented.append(result)
                
        # Temporal augmentation
        if "temporal" in augment_types:
            temporal_results = self.temporal.augment(
                pose, timestamps, actions
            )
            for result in temporal_results:
                result.source_episode_id = source_episode_id
                # 액션도 함께 재계산 필요
                if actions is not None and len(result.pose) > 1:
                    # 간단히 포즈 차분으로 재계산
                    result.actions = self._recompute_actions(result.pose)
                augmented.append(result)
                
        # Viewpoint augmentation (3D 데이터에만)
        if "viewpoint" in augment_types:
            viewpoint_results = self.viewpoint.augment(pose, timestamps)
            for result in viewpoint_results:
                result.source_episode_id = source_episode_id
                augmented.append(result)
                
        logger.info(
            f"Augmentation complete: {len(augmented)} variants from "
            f"source={source_episode_id or 'unknown'}"
        )
        
        return augmented
    
    def _recompute_actions(self, pose: np.ndarray) -> np.ndarray:
        """포즈에서 액션 재계산 (간단한 position delta)"""
        # 손목 인덱스
        LEFT_WRIST = MEDIAPIPE_LANDMARKS.get("LEFT_WRIST", 15)
        RIGHT_WRIST = MEDIAPIPE_LANDMARKS.get("RIGHT_WRIST", 16)
        
        J = pose.shape[1]
        if LEFT_WRIST < J and RIGHT_WRIST < J:
            eef = np.stack([pose[:, LEFT_WRIST], pose[:, RIGHT_WRIST]], axis=1)
        else:
            eef = pose[:, :2, :]  # 첫 2개 관절 사용
            
        delta = eef[1:] - eef[:-1]
        return delta.reshape(len(delta), -1)
    
    def augment_batch(
        self,
        episodes: List[Dict[str, Any]],
        augment_ratio: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        배치 증강 (일부 에피소드만)
        
        Args:
            episodes: 에피소드 리스트
            augment_ratio: 증강할 에피소드 비율
            
        Returns:
            증강된 에피소드 포함 전체 리스트
        """
        result = episodes.copy()
        
        n_augment = int(len(episodes) * augment_ratio)
        indices = np.random.choice(len(episodes), n_augment, replace=False)
        
        for idx in indices:
            ep = episodes[idx]
            aug_results = self.augment(
                pose=ep["pose"],
                timestamps=ep.get("timestamps", np.arange(len(ep["pose"])) / 30.0),
                actions=ep.get("actions"),
                source_episode_id=ep.get("episode_id", f"ep_{idx}"),
                **kwargs,
            )
            
            for aug in aug_results:
                result.append({
                    "pose": aug.pose,
                    "timestamps": aug.timestamps,
                    "actions": aug.actions,
                    "episode_id": aug.aug_id,
                    "metadata": {
                        "source_episode_id": aug.source_episode_id,
                        "aug_type": aug.aug_type,
                        "aug_params": aug.aug_params,
                    },
                })
                
        logger.info(
            f"Batch augmentation: {len(episodes)} → {len(result)} episodes "
            f"(+{len(result) - len(episodes)} augmented)"
        )
        
        return result
