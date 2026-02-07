"""
FR-4.1: Coordinate Normalization

좌표계 정규화 및 변환
- Reference Frame Alignment (골반 중심 기준 좌표계)
- Relative Coordinate Computation (상대 좌표)
- Dimensionality Reduction (차원 축소)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation

from core.logging_config import setup_logger
from transformation.spec import (
    TransformConfig,
    MEDIAPIPE_LANDMARKS,
    SKELETON_PARENT,
    get_joint_indices,
)

logger = setup_logger(__name__)


@dataclass
class FrameTransform:
    """프레임별 변환 정보"""
    origin: np.ndarray  # [3] 원점 (pelvis 위치)
    rotation: np.ndarray  # [3, 3] 회전 행렬
    scale: float  # 스케일 팩터
    valid: bool = True  # 유효 여부


@dataclass
class AlignmentResult:
    """정렬 결과"""
    pose_aligned: np.ndarray  # [T, J, 3]
    transforms: List[FrameTransform]  # 프레임별 변환
    valid_mask: np.ndarray  # [T] 유효 프레임 마스크
    quality_metrics: Dict[str, float]  # 품질 메트릭


class ReferenceFrameAligner:
    """
    Task 4.1.1: Reference Frame Alignment
    
    골반 중심 기준 좌표계 설정, 회전/스케일 불변 변환
    """
    
    # 핵심 관절 인덱스
    LEFT_HIP = MEDIAPIPE_LANDMARKS["LEFT_HIP"]
    RIGHT_HIP = MEDIAPIPE_LANDMARKS["RIGHT_HIP"]
    LEFT_SHOULDER = MEDIAPIPE_LANDMARKS["LEFT_SHOULDER"]
    RIGHT_SHOULDER = MEDIAPIPE_LANDMARKS["RIGHT_SHOULDER"]
    NOSE = MEDIAPIPE_LANDMARKS["NOSE"]
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.eps = 1e-8
        
    def align(
        self,
        pose_3d: np.ndarray,
        pose_conf: Optional[np.ndarray] = None,
    ) -> AlignmentResult:
        """
        좌표계 정렬 수행
        
        Args:
            pose_3d: [T, J, 3] 3D 포즈 좌표
            pose_conf: [T, J] 또는 [T] 신뢰도
            
        Returns:
            AlignmentResult: 정렬된 포즈와 변환 정보
        """
        T, J, _ = pose_3d.shape
        
        # 신뢰도 처리
        if pose_conf is None:
            pose_conf = np.ones((T, J))
        elif pose_conf.ndim == 1:
            pose_conf = np.tile(pose_conf[:, np.newaxis], (1, J))
            
        # 결과 배열 초기화
        pose_aligned = np.zeros_like(pose_3d)
        transforms = []
        valid_mask = np.zeros(T, dtype=bool)
        
        # 이전 유효 변환 (hold 정책용)
        last_valid_transform = None
        
        for t in range(T):
            frame_pose = pose_3d[t]  # [J, 3]
            frame_conf = pose_conf[t]  # [J]
            
            # 유효성 검사
            is_valid = self._check_frame_validity(frame_conf)
            
            if is_valid:
                transform = self._compute_frame_transform(frame_pose, frame_conf)
                if transform.valid:
                    last_valid_transform = transform
            else:
                # 이전 유효 변환 사용 (hold 정책)
                if last_valid_transform is not None:
                    transform = FrameTransform(
                        origin=last_valid_transform.origin.copy(),
                        rotation=last_valid_transform.rotation.copy(),
                        scale=last_valid_transform.scale,
                        valid=False,
                    )
                else:
                    transform = FrameTransform(
                        origin=np.zeros(3),
                        rotation=np.eye(3),
                        scale=1.0,
                        valid=False,
                    )
            
            # 변환 적용
            aligned = self._apply_transform(frame_pose, transform)
            pose_aligned[t] = aligned
            transforms.append(transform)
            valid_mask[t] = transform.valid
            
        # 품질 메트릭 계산
        quality_metrics = self._compute_quality_metrics(pose_aligned, valid_mask, transforms)
        
        logger.info(
            f"Reference frame alignment: {valid_mask.sum()}/{T} valid frames, "
            f"scale range: [{quality_metrics.get('scale_min', 0):.3f}, {quality_metrics.get('scale_max', 0):.3f}]"
        )
        
        return AlignmentResult(
            pose_aligned=pose_aligned,
            transforms=transforms,
            valid_mask=valid_mask,
            quality_metrics=quality_metrics,
        )
    
    def _check_frame_validity(self, conf: np.ndarray) -> bool:
        """프레임 유효성 검사"""
        # 핵심 관절 신뢰도 확인
        core_joints = [self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_SHOULDER, self.RIGHT_SHOULDER]
        core_conf = conf[core_joints]
        return np.all(core_conf >= self.config.conf_min)
    
    def _compute_frame_transform(
        self,
        pose: np.ndarray,
        conf: np.ndarray,
    ) -> FrameTransform:
        """프레임별 변환 계산"""
        try:
            # 1. Origin: Pelvis center
            left_hip = pose[self.LEFT_HIP]
            right_hip = pose[self.RIGHT_HIP]
            origin = (left_hip + right_hip) / 2
            
            # 2. Rotation: 바디 좌표계 구성
            left_shoulder = pose[self.LEFT_SHOULDER]
            right_shoulder = pose[self.RIGHT_SHOULDER]
            
            # X축: 오른쪽 방향
            x_axis = right_shoulder - left_shoulder
            x_norm = np.linalg.norm(x_axis)
            if x_norm < self.eps:
                return FrameTransform(origin, np.eye(3), 1.0, False)
            x_axis = x_axis / x_norm
            
            # Y축: 위쪽 방향 (어깨 중심 - pelvis)
            shoulder_center = (left_shoulder + right_shoulder) / 2
            y_axis = shoulder_center - origin
            y_norm = np.linalg.norm(y_axis)
            if y_norm < self.eps:
                return FrameTransform(origin, np.eye(3), 1.0, False)
            y_axis = y_axis / y_norm
            
            # Z축: forward (x cross y)
            z_axis = np.cross(x_axis, y_axis)
            z_norm = np.linalg.norm(z_axis)
            if z_norm < self.eps:
                return FrameTransform(origin, np.eye(3), 1.0, False)
            z_axis = z_axis / z_norm
            
            # Y축 재계산 (직교화)
            y_axis = np.cross(z_axis, x_axis)
            
            # 회전 행렬 구성 (행 벡터)
            rotation = np.array([x_axis, y_axis, z_axis])
            
            # 3. Scale: 어깨 너비
            if self.config.normalization.scale == "shoulder_width":
                scale = np.linalg.norm(right_shoulder - left_shoulder)
            elif self.config.normalization.scale == "hip_width":
                scale = np.linalg.norm(right_hip - left_hip)
            elif self.config.normalization.scale == "torso_length":
                scale = np.linalg.norm(shoulder_center - origin)
            else:
                scale = 1.0
                
            if scale < self.eps:
                scale = 1.0
                
            return FrameTransform(origin, rotation, scale, True)
            
        except Exception as e:
            logger.warning(f"Frame transform computation failed: {e}")
            return FrameTransform(np.zeros(3), np.eye(3), 1.0, False)
    
    def _apply_transform(
        self,
        pose: np.ndarray,
        transform: FrameTransform,
    ) -> np.ndarray:
        """변환 적용"""
        # Translation 제거
        centered = pose - transform.origin
        
        # Rotation 적용
        if self.config.normalization.rotation != "none":
            rotated = centered @ transform.rotation.T
        else:
            rotated = centered
            
        # Scale 정규화
        if self.config.normalization.scale != "none":
            scaled = rotated / (transform.scale + self.eps)
        else:
            scaled = rotated
            
        return scaled
    
    def _compute_quality_metrics(
        self,
        pose_aligned: np.ndarray,
        valid_mask: np.ndarray,
        transforms: List[FrameTransform],
    ) -> Dict[str, float]:
        """품질 메트릭 계산"""
        metrics = {}
        
        # 유효 프레임 비율
        metrics["valid_ratio"] = valid_mask.sum() / len(valid_mask)
        
        # 스케일 범위
        scales = [t.scale for t in transforms if t.valid]
        if scales:
            metrics["scale_min"] = min(scales)
            metrics["scale_max"] = max(scales)
            metrics["scale_mean"] = np.mean(scales)
            metrics["scale_std"] = np.std(scales)
        
        # 정렬 후 범위 확인 (이상치 탐지)
        if valid_mask.sum() > 0:
            valid_poses = pose_aligned[valid_mask]
            metrics["coord_max"] = np.abs(valid_poses).max()
            metrics["coord_mean"] = np.abs(valid_poses).mean()
            
        return metrics


class RelativeCoordinateComputer:
    """
    Task 4.1.2: Relative Coordinate Computation
    
    Joint-to-Joint 상대 좌표, End-Effector 중심 표현
    """
    
    LEFT_WRIST = MEDIAPIPE_LANDMARKS["LEFT_WRIST"]
    RIGHT_WRIST = MEDIAPIPE_LANDMARKS["RIGHT_WRIST"]
    LEFT_HIP = MEDIAPIPE_LANDMARKS["LEFT_HIP"]
    RIGHT_HIP = MEDIAPIPE_LANDMARKS["RIGHT_HIP"]
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.eps = 1e-8
        
    def compute_joint_relative(
        self,
        pose: np.ndarray,
        method: str = "parent",
        reference_joint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Joint-to-Joint 상대 좌표 계산
        
        Args:
            pose: [T, J, 3] 정규화된 포즈
            method: "parent" (부모 관절 기준) 또는 "reference" (기준 관절)
            reference_joint: method="reference"일 때 기준 관절 이름
            
        Returns:
            [T, J, 3] 상대 좌표
        """
        T, J, _ = pose.shape
        relative = np.zeros_like(pose)
        
        if method == "parent":
            # 부모 관절 기준 상대 좌표
            for joint_name, parent_name in SKELETON_PARENT.items():
                if joint_name not in MEDIAPIPE_LANDMARKS:
                    continue
                j_idx = MEDIAPIPE_LANDMARKS[joint_name]
                if j_idx >= J:
                    continue
                    
                if parent_name is None:
                    # 루트 관절은 절대 좌표 유지
                    relative[:, j_idx] = pose[:, j_idx]
                else:
                    p_idx = MEDIAPIPE_LANDMARKS.get(parent_name, 0)
                    if p_idx < J:
                        relative[:, j_idx] = pose[:, j_idx] - pose[:, p_idx]
                    else:
                        relative[:, j_idx] = pose[:, j_idx]
                        
        elif method == "reference":
            # 기준 관절 기준 상대 좌표
            if reference_joint is None:
                reference_joint = "LEFT_HIP"  # 기본: pelvis 근처
            ref_idx = MEDIAPIPE_LANDMARKS.get(reference_joint, self.LEFT_HIP)
            ref_pos = pose[:, ref_idx:ref_idx+1, :]  # [T, 1, 3]
            relative = pose - ref_pos
            
        return relative
    
    def compute_eef_positions(
        self,
        pose: np.ndarray,
        hands_3d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        End-Effector 위치 추출
        
        Args:
            pose: [T, J, 3] 포즈
            hands_3d: [T, H, 3] 손 랜드마크 (있으면)
            
        Returns:
            [T, 2, 3] 양손 EEF 위치
        """
        T = pose.shape[0]
        eef = np.zeros((T, 2, 3))
        
        if hands_3d is not None and hands_3d.shape[1] >= 42:
            # 손 중심 사용 (21개 랜드마크 x 2)
            left_hand = hands_3d[:, :21, :].mean(axis=1)
            right_hand = hands_3d[:, 21:42, :].mean(axis=1)
            eef[:, 0] = left_hand
            eef[:, 1] = right_hand
        else:
            # 손목 사용
            eef[:, 0] = pose[:, self.LEFT_WRIST]
            eef[:, 1] = pose[:, self.RIGHT_WRIST]
            
        return eef
    
    def compute_object_relative(
        self,
        eef_pos: np.ndarray,
        obj_bbox: Optional[np.ndarray] = None,
        obj_pos_3d: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        객체 상대 위치 계산
        
        Args:
            eef_pos: [T, 2, 3] EEF 위치
            obj_bbox: [T, K, 4] 2D bbox (cx, cy, w, h)
            obj_pos_3d: [T, K, 3] 3D 객체 위치 (있으면)
            
        Returns:
            obj_rel: [T, K, 3] 객체 상대 위치
            obj_mask: [T, K] 유효 객체 마스크
        """
        T = eef_pos.shape[0]
        
        if obj_pos_3d is not None:
            K = obj_pos_3d.shape[1]
            # 3D 상대 위치 (왼손 기준)
            eef_left = eef_pos[:, 0:1, :]  # [T, 1, 3]
            obj_rel = obj_pos_3d - eef_left
            obj_mask = ~np.isnan(obj_pos_3d[:, :, 0])
            
        elif obj_bbox is not None:
            K = obj_bbox.shape[1]
            # 2D 관계로 저장 (cx, cy, size)
            obj_rel = np.zeros((T, K, 3))
            obj_rel[:, :, 0] = obj_bbox[:, :, 0]  # cx
            obj_rel[:, :, 1] = obj_bbox[:, :, 1]  # cy
            obj_rel[:, :, 2] = np.sqrt(obj_bbox[:, :, 2] * obj_bbox[:, :, 3])  # sqrt(w*h)
            obj_mask = ~np.isnan(obj_bbox[:, :, 0])
            
        else:
            # 객체 없음
            K = 1
            obj_rel = np.zeros((T, K, 3))
            obj_mask = np.zeros((T, K), dtype=bool)
            
        return obj_rel, obj_mask


class DimensionalityReducer:
    """
    Task 4.1.3: Dimensionality Reduction
    
    중요 관절 선택, PCA 적용, 데이터 압축
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.pca_model = None
        self.joint_indices = None
        
    def select_joints(
        self,
        pose: np.ndarray,
        joints_keep: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        중요 관절 선택
        
        Args:
            pose: [T, J, 3] 전체 포즈
            joints_keep: 유지할 관절 이름 리스트
            
        Returns:
            [T, J_keep, 3] 선택된 관절만
        """
        if joints_keep is None:
            joints_keep = self.config.joints_keep
            
        self.joint_indices = get_joint_indices(joints_keep)
        return pose[:, self.joint_indices, :]
    
    def fit_pca(
        self,
        poses: np.ndarray,
        n_components: Optional[int] = None,
    ) -> None:
        """
        PCA 모델 학습
        
        Args:
            poses: [N, T, J, 3] 여러 에피소드의 포즈
            n_components: PCA 차원 수
        """
        if n_components is None:
            n_components = self.config.pca_components
            
        # Flatten: [N*T, J*3]
        if poses.ndim == 4:
            N, T, J, _ = poses.shape
            X = poses.reshape(N * T, -1)
        else:
            T, J, _ = poses.shape
            X = poses.reshape(T, -1)
            
        # PCA 학습
        from sklearn.decomposition import PCA
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(X)
        
        logger.info(
            f"PCA fitted: {n_components} components, "
            f"explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}"
        )
    
    def apply_pca(self, pose: np.ndarray) -> np.ndarray:
        """
        PCA 변환 적용
        
        Args:
            pose: [T, J, 3] 포즈
            
        Returns:
            [T, D] PCA 변환된 데이터
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca first.")
            
        T, J, _ = pose.shape
        X = pose.reshape(T, -1)
        return self.pca_model.transform(X)
    
    def compress_dtype(
        self,
        data: np.ndarray,
        dtype: str = "float16",
    ) -> np.ndarray:
        """
        dtype 압축
        
        Args:
            data: 입력 데이터
            dtype: 출력 dtype ("float16" 또는 "float32")
            
        Returns:
            압축된 데이터
        """
        if dtype == "float16":
            return data.astype(np.float16)
        elif dtype == "float32":
            return data.astype(np.float32)
        else:
            return data
    
    def save_pca_model(self, path: str) -> None:
        """PCA 모델 저장"""
        if self.pca_model is None:
            raise ValueError("No PCA model to save")
            
        import joblib
        joblib.dump({
            "pca": self.pca_model,
            "joint_indices": self.joint_indices,
        }, path)
        logger.info(f"PCA model saved to {path}")
    
    def load_pca_model(self, path: str) -> None:
        """PCA 모델 로드"""
        import joblib
        data = joblib.load(path)
        self.pca_model = data["pca"]
        self.joint_indices = data["joint_indices"]
        logger.info(f"PCA model loaded from {path}")


class CoordinateNormalizer:
    """
    좌표 정규화 통합 파이프라인
    
    Reference Frame Alignment + Relative Coordinates + Dimensionality Reduction
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.aligner = ReferenceFrameAligner(config)
        self.relative_computer = RelativeCoordinateComputer(config)
        self.reducer = DimensionalityReducer(config)
        
    def normalize(
        self,
        pose_3d: np.ndarray,
        pose_conf: Optional[np.ndarray] = None,
        hands_3d: Optional[np.ndarray] = None,
        objects: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        전체 정규화 파이프라인 실행
        
        Args:
            pose_3d: [T, J, 3] 원본 3D 포즈
            pose_conf: [T, J] 신뢰도
            hands_3d: [T, H, 3] 손 랜드마크
            objects: {"bbox": [T,K,4], "pos_3d": [T,K,3], ...}
            
        Returns:
            정규화된 데이터 딕셔너리
        """
        result = {}
        
        # 1. Reference Frame Alignment
        alignment = self.aligner.align(pose_3d, pose_conf)
        result["pose_aligned"] = alignment.pose_aligned
        result["valid_mask"] = alignment.valid_mask
        result["transforms"] = alignment.transforms
        result["quality_metrics"] = alignment.quality_metrics
        
        # 2. Joint Selection (Dimensionality Reduction 1단계)
        pose_selected = self.reducer.select_joints(alignment.pose_aligned)
        result["pose_selected"] = pose_selected
        
        # 3. Relative Coordinates (선택적)
        result["joint_relative"] = self.relative_computer.compute_joint_relative(
            alignment.pose_aligned, method="parent"
        )
        
        # 4. EEF Positions
        eef_pos = self.relative_computer.compute_eef_positions(
            alignment.pose_aligned, hands_3d
        )
        result["eef_positions"] = eef_pos
        
        # 5. Object Relations
        if objects:
            obj_rel, obj_mask = self.relative_computer.compute_object_relative(
                eef_pos,
                obj_bbox=objects.get("bbox"),
                obj_pos_3d=objects.get("pos_3d"),
            )
            result["obj_relative"] = obj_rel
            result["obj_mask"] = obj_mask
            
        # 6. PCA (선택적)
        if self.config.pca_enable and self.reducer.pca_model is not None:
            result["pose_pca"] = self.reducer.apply_pca(pose_selected)
            
        # 7. dtype 압축
        result["pose_compressed"] = self.reducer.compress_dtype(
            pose_selected, self.config.dtype_out
        )
        
        logger.info(
            f"Coordinate normalization complete: "
            f"input {pose_3d.shape} → output {result['pose_compressed'].shape}"
        )
        
        return result
