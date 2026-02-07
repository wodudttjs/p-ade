"""
Transformation Specification and Configuration

State/Action 스키마, 버전 관리, 설정 관리
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import hashlib
import json


VERSION = "4.0.0"


class NormalizationMode(Enum):
    """정규화 모드"""
    PELVIS = "pelvis"
    TORSO = "torso"
    NONE = "none"


class ScaleReference(Enum):
    """스케일 기준"""
    SHOULDER_WIDTH = "shoulder_width"
    HIP_WIDTH = "hip_width"
    TORSO_LENGTH = "torso_length"
    NONE = "none"


class RotationReference(Enum):
    """회전 기준"""
    SHOULDERS = "shoulders"
    HIPS = "hips"
    SPINE = "spine"
    NONE = "none"


class SmoothingMethod(Enum):
    """스무딩 방법"""
    SAVGOL = "savgol"
    MOVING_AVERAGE = "moving_average"
    GAUSSIAN = "gaussian"
    NONE = "none"


# MediaPipe Pose Landmark 인덱스
MEDIAPIPE_LANDMARKS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# 좌우 대칭 관절 매핑 (미러링용)
LEFT_RIGHT_PAIRS = [
    ("LEFT_EYE_INNER", "RIGHT_EYE_INNER"),
    ("LEFT_EYE", "RIGHT_EYE"),
    ("LEFT_EYE_OUTER", "RIGHT_EYE_OUTER"),
    ("LEFT_EAR", "RIGHT_EAR"),
    ("MOUTH_LEFT", "MOUTH_RIGHT"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_PINKY", "RIGHT_PINKY"),
    ("LEFT_INDEX", "RIGHT_INDEX"),
    ("LEFT_THUMB", "RIGHT_THUMB"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_KNEE", "RIGHT_KNEE"),
    ("LEFT_ANKLE", "RIGHT_ANKLE"),
    ("LEFT_HEEL", "RIGHT_HEEL"),
    ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
]

# 기본 유지 관절 (손목, 팔꿈치, 어깨, 엉덩이)
DEFAULT_JOINTS_KEEP = [
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
]

# 골격 트리 (부모 관절 매핑)
SKELETON_PARENT = {
    "NOSE": None,
    "LEFT_EYE_INNER": "NOSE",
    "LEFT_EYE": "LEFT_EYE_INNER",
    "LEFT_EYE_OUTER": "LEFT_EYE",
    "RIGHT_EYE_INNER": "NOSE",
    "RIGHT_EYE": "RIGHT_EYE_INNER",
    "RIGHT_EYE_OUTER": "RIGHT_EYE",
    "LEFT_EAR": "LEFT_EYE_OUTER",
    "RIGHT_EAR": "RIGHT_EYE_OUTER",
    "MOUTH_LEFT": "NOSE",
    "MOUTH_RIGHT": "NOSE",
    "LEFT_SHOULDER": "NOSE",
    "RIGHT_SHOULDER": "NOSE",
    "LEFT_ELBOW": "LEFT_SHOULDER",
    "RIGHT_ELBOW": "RIGHT_SHOULDER",
    "LEFT_WRIST": "LEFT_ELBOW",
    "RIGHT_WRIST": "RIGHT_ELBOW",
    "LEFT_PINKY": "LEFT_WRIST",
    "RIGHT_PINKY": "RIGHT_WRIST",
    "LEFT_INDEX": "LEFT_WRIST",
    "RIGHT_INDEX": "RIGHT_WRIST",
    "LEFT_THUMB": "LEFT_WRIST",
    "RIGHT_THUMB": "RIGHT_WRIST",
    "LEFT_HIP": "NOSE",
    "RIGHT_HIP": "NOSE",
    "LEFT_KNEE": "LEFT_HIP",
    "RIGHT_KNEE": "RIGHT_HIP",
    "LEFT_ANKLE": "LEFT_KNEE",
    "RIGHT_ANKLE": "RIGHT_KNEE",
    "LEFT_HEEL": "LEFT_ANKLE",
    "RIGHT_HEEL": "RIGHT_ANKLE",
    "LEFT_FOOT_INDEX": "LEFT_ANKLE",
    "RIGHT_FOOT_INDEX": "RIGHT_ANKLE",
}


@dataclass
class SmoothingConfig:
    """스무딩 설정"""
    method: str = "savgol"
    window: int = 11
    polyorder: int = 3


@dataclass
class NormalizationConfig:
    """정규화 설정"""
    mode: str = "pelvis"
    scale: str = "shoulder_width"
    rotation: str = "shoulders"


@dataclass
class SpatialAugmentConfig:
    """공간 증강 설정"""
    enable_mirror: bool = True
    enable_scale: bool = True
    scale_range: tuple = (0.9, 1.1)
    enable_noise: bool = True
    noise_sigma: float = 0.01


@dataclass
class TemporalAugmentConfig:
    """시간 증강 설정"""
    enable_speed: bool = True
    speed_range: tuple = (0.8, 1.2)
    enable_offset: bool = True
    max_offset_sec: float = 0.5
    enable_crop: bool = True
    min_duration_sec: float = 1.0


@dataclass
class ViewpointAugmentConfig:
    """시점 증강 설정"""
    enable_rotation: bool = True
    yaw_range: tuple = (-15.0, 15.0)
    pitch_range: tuple = (-10.0, 10.0)
    roll_range: tuple = (-5.0, 5.0)
    enable_depth_normalize: bool = True


@dataclass
class AugmentConfig:
    """전체 증강 설정"""
    enable: bool = True
    spatial: SpatialAugmentConfig = field(default_factory=SpatialAugmentConfig)
    temporal: TemporalAugmentConfig = field(default_factory=TemporalAugmentConfig)
    viewpoint: ViewpointAugmentConfig = field(default_factory=ViewpointAugmentConfig)


@dataclass
class TransformConfig:
    """전체 변환 설정"""
    # 기본 설정
    target_fps: int = 30
    conf_min: float = 0.5
    gap_max_frames: int = 10
    
    # 스무딩
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    
    # 정규화
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    # 관절 선택
    joints_keep: List[str] = field(default_factory=lambda: DEFAULT_JOINTS_KEEP.copy())
    
    # PCA 설정
    pca_enable: bool = False
    pca_components: int = 20
    
    # 출력 dtype
    dtype_out: str = "float16"
    
    # 증강
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    
    # 버전
    version: str = VERSION
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "target_fps": self.target_fps,
            "conf_min": self.conf_min,
            "gap_max_frames": self.gap_max_frames,
            "smoothing": {
                "method": self.smoothing.method,
                "window": self.smoothing.window,
                "polyorder": self.smoothing.polyorder,
            },
            "normalization": {
                "mode": self.normalization.mode,
                "scale": self.normalization.scale,
                "rotation": self.normalization.rotation,
            },
            "joints_keep": self.joints_keep,
            "pca_enable": self.pca_enable,
            "pca_components": self.pca_components,
            "dtype_out": self.dtype_out,
            "augment": {
                "enable": self.augment.enable,
                "spatial": {
                    "enable_mirror": self.augment.spatial.enable_mirror,
                    "enable_scale": self.augment.spatial.enable_scale,
                    "scale_range": self.augment.spatial.scale_range,
                    "enable_noise": self.augment.spatial.enable_noise,
                    "noise_sigma": self.augment.spatial.noise_sigma,
                },
                "temporal": {
                    "enable_speed": self.augment.temporal.enable_speed,
                    "speed_range": self.augment.temporal.speed_range,
                    "enable_offset": self.augment.temporal.enable_offset,
                    "max_offset_sec": self.augment.temporal.max_offset_sec,
                    "enable_crop": self.augment.temporal.enable_crop,
                    "min_duration_sec": self.augment.temporal.min_duration_sec,
                },
                "viewpoint": {
                    "enable_rotation": self.augment.viewpoint.enable_rotation,
                    "yaw_range": self.augment.viewpoint.yaw_range,
                    "pitch_range": self.augment.viewpoint.pitch_range,
                    "roll_range": self.augment.viewpoint.roll_range,
                    "enable_depth_normalize": self.augment.viewpoint.enable_depth_normalize,
                },
            },
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformConfig":
        """딕셔너리에서 설정 생성"""
        config = cls()
        
        if "target_fps" in data:
            config.target_fps = data["target_fps"]
        if "conf_min" in data:
            config.conf_min = data["conf_min"]
        if "gap_max_frames" in data:
            config.gap_max_frames = data["gap_max_frames"]
        
        if "smoothing" in data:
            sm = data["smoothing"]
            config.smoothing = SmoothingConfig(
                method=sm.get("method", "savgol"),
                window=sm.get("window", 11),
                polyorder=sm.get("polyorder", 3),
            )
        
        if "normalization" in data:
            nm = data["normalization"]
            config.normalization = NormalizationConfig(
                mode=nm.get("mode", "pelvis"),
                scale=nm.get("scale", "shoulder_width"),
                rotation=nm.get("rotation", "shoulders"),
            )
        
        if "joints_keep" in data:
            config.joints_keep = data["joints_keep"]
        if "pca_enable" in data:
            config.pca_enable = data["pca_enable"]
        if "pca_components" in data:
            config.pca_components = data["pca_components"]
        if "dtype_out" in data:
            config.dtype_out = data["dtype_out"]
        
        if "augment" in data:
            aug = data["augment"]
            config.augment = AugmentConfig(
                enable=aug.get("enable", True),
                spatial=SpatialAugmentConfig(**aug.get("spatial", {})) if "spatial" in aug else SpatialAugmentConfig(),
                temporal=TemporalAugmentConfig(**aug.get("temporal", {})) if "temporal" in aug else TemporalAugmentConfig(),
                viewpoint=ViewpointAugmentConfig(**aug.get("viewpoint", {})) if "viewpoint" in aug else ViewpointAugmentConfig(),
            )
        
        if "version" in data:
            config.version = data["version"]
        
        return config
    
    def get_params_hash(self) -> str:
        """설정 해시 생성 (재현성용)"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class StateSpec:
    """State 스키마 정의"""
    # 관절 위치
    joint_positions: bool = True
    num_joints: int = 8  # joints_keep 개수
    
    # 관절 속도
    joint_velocities: bool = True
    
    # 객체 관계
    object_relations: bool = False
    max_objects: int = 5
    
    # 추가 features
    confidence_stats: bool = True
    
    def get_state_dim(self) -> int:
        """State 차원 계산"""
        dim = 0
        if self.joint_positions:
            dim += self.num_joints * 3
        if self.joint_velocities:
            dim += self.num_joints * 3
        if self.object_relations:
            dim += self.max_objects * 3
        if self.confidence_stats:
            dim += 2  # mean, min
        return dim
    
    def to_string(self) -> str:
        """스키마 문자열 생성"""
        parts = []
        if self.joint_positions:
            parts.append(f"pos[{self.num_joints}x3]")
        if self.joint_velocities:
            parts.append(f"vel[{self.num_joints}x3]")
        if self.object_relations:
            parts.append(f"obj[{self.max_objects}x3]")
        if self.confidence_stats:
            parts.append("conf[2]")
        return "+".join(parts)


@dataclass
class ActionSpec:
    """Action 스키마 정의"""
    # Position delta
    position_delta: bool = True
    eef_only: bool = True  # True면 손목만, False면 전체 관절
    num_eef: int = 2  # 양손
    
    # Rotation delta
    rotation_delta: bool = False
    rotation_repr: str = "quaternion"  # quaternion(4D) or axis_angle(3D)
    
    # Gripper state
    gripper_state: bool = False
    
    def get_action_dim(self) -> int:
        """Action 차원 계산"""
        dim = 0
        if self.position_delta:
            if self.eef_only:
                dim += self.num_eef * 3
            else:
                dim += 8 * 3  # 기본 8개 관절
        if self.rotation_delta:
            if self.rotation_repr == "quaternion":
                dim += self.num_eef * 4
            else:
                dim += self.num_eef * 3
        if self.gripper_state:
            dim += self.num_eef
        return dim
    
    def to_string(self) -> str:
        """스키마 문자열 생성"""
        parts = []
        if self.position_delta:
            if self.eef_only:
                parts.append(f"dpos_eef[{self.num_eef}x3]")
            else:
                parts.append("dpos_joints[8x3]")
        if self.rotation_delta:
            if self.rotation_repr == "quaternion":
                parts.append(f"drot_quat[{self.num_eef}x4]")
            else:
                parts.append(f"drot_aa[{self.num_eef}x3]")
        if self.gripper_state:
            parts.append(f"grip[{self.num_eef}]")
        return "+".join(parts)


def get_joint_indices(joint_names: List[str]) -> List[int]:
    """관절 이름에서 인덱스 추출"""
    return [MEDIAPIPE_LANDMARKS[name] for name in joint_names if name in MEDIAPIPE_LANDMARKS]


def get_mirror_swap_indices(joint_names: List[str]) -> Dict[int, int]:
    """미러링을 위한 좌우 인덱스 스왑 맵 생성"""
    swap_map = {}
    for left, right in LEFT_RIGHT_PAIRS:
        if left in joint_names and right in joint_names:
            left_idx = joint_names.index(left)
            right_idx = joint_names.index(right)
            swap_map[left_idx] = right_idx
            swap_map[right_idx] = left_idx
    return swap_map


def create_default_config() -> TransformConfig:
    """기본 설정 생성"""
    return TransformConfig()


def load_config_from_yaml(yaml_path: str) -> TransformConfig:
    """YAML 파일에서 설정 로드"""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TransformConfig.from_dict(data)


def save_config_to_yaml(config: TransformConfig, yaml_path: str) -> None:
    """설정을 YAML 파일로 저장"""
    import yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
