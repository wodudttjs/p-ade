"""
FR-4.2: Action Encoding

State-Action 쌍 생성
- State Representation (관절 위치, 속도, 객체 관계)
- Action Computation (Position delta, Rotation delta, Gripper state)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation

from core.logging_config import setup_logger
from transformation.spec import (
    TransformConfig,
    StateSpec,
    ActionSpec,
    MEDIAPIPE_LANDMARKS,
)

logger = setup_logger(__name__)


@dataclass
class StateActionPair:
    """State-Action 쌍"""
    states: np.ndarray  # [T, S]
    actions: np.ndarray  # [T-1, A]
    timestamps: np.ndarray  # [T]
    masks: np.ndarray  # [T]
    state_spec: StateSpec
    action_spec: ActionSpec
    metadata: Dict[str, Any]


class StateBuilder:
    """
    Task 4.2.1: State Representation
    
    관절 위치, 속도, 객체 관계를 포함한 상태 벡터 생성
    """
    
    def __init__(
        self,
        config: Optional[TransformConfig] = None,
        state_spec: Optional[StateSpec] = None,
    ):
        self.config = config or TransformConfig()
        self.state_spec = state_spec or StateSpec()
        
    def build_state(
        self,
        pose: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        obj_rel: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        상태 벡터 생성
        
        Args:
            pose: [T, J, 3] 정규화된 포즈
            velocity: [T, J, 3] 속도 (선택)
            obj_rel: [T, K, 3] 객체 상대 위치 (선택)
            conf: [T, J] 신뢰도 (선택)
            masks: [T] 유효 마스크 (선택)
            
        Returns:
            states: [T, S] 상태 벡터
            masks: [T] 상태 마스크
        """
        T = pose.shape[0]
        state_parts = []
        
        # 1. Joint positions
        if self.state_spec.joint_positions:
            pose_flat = pose.reshape(T, -1)
            state_parts.append(pose_flat)
            
        # 2. Joint velocities
        if self.state_spec.joint_velocities:
            if velocity is None:
                # 속도가 없으면 0으로
                velocity = np.zeros_like(pose)
            vel_flat = velocity.reshape(T, -1)
            state_parts.append(vel_flat)
            
        # 3. Object relations
        if self.state_spec.object_relations and obj_rel is not None:
            K = obj_rel.shape[1]
            max_k = self.state_spec.max_objects
            
            if K > max_k:
                # Top-K 선택 (가장 가까운 것)
                distances = np.linalg.norm(obj_rel, axis=2)
                top_k_indices = np.argsort(distances, axis=1)[:, :max_k]
                obj_selected = np.take_along_axis(
                    obj_rel,
                    top_k_indices[:, :, np.newaxis],
                    axis=1
                )
            elif K < max_k:
                # Zero padding
                obj_selected = np.zeros((T, max_k, 3))
                obj_selected[:, :K, :] = obj_rel
            else:
                obj_selected = obj_rel
                
            obj_flat = obj_selected.reshape(T, -1)
            state_parts.append(obj_flat)
            
        # 4. Confidence statistics
        if self.state_spec.confidence_stats and conf is not None:
            if conf.ndim == 1:
                conf_stats = np.stack([conf, conf], axis=1)
            else:
                conf_mean = conf.mean(axis=1, keepdims=True)
                conf_min = conf.min(axis=1, keepdims=True)
                conf_stats = np.concatenate([conf_mean, conf_min], axis=1)
            state_parts.append(conf_stats)
            
        # 상태 벡터 결합
        if state_parts:
            states = np.concatenate(state_parts, axis=1)
        else:
            states = pose.reshape(T, -1)
            
        # 마스크 처리
        if masks is None:
            masks = np.ones(T, dtype=bool)
            
        logger.debug(f"Built states: shape={states.shape}, spec={self.state_spec.to_string()}")
        
        return states, masks
    
    def build_state_tensor(
        self,
        pose: np.ndarray,
        velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        텐서 형태 상태 (연구/해석용)
        
        Args:
            pose: [T, J, 3] 포즈
            velocity: [T, J, 3] 속도
            
        Returns:
            [T, J, C] 상태 텐서 (C는 채널 수)
        """
        T, J, _ = pose.shape
        channels = [pose]
        
        if velocity is not None:
            channels.append(velocity)
            
        state_tensor = np.concatenate(channels, axis=2)
        return state_tensor


class ActionComputer:
    """
    Task 4.2.2: Action Computation
    
    Position delta, Rotation delta, Gripper state 계산
    """
    
    LEFT_WRIST = MEDIAPIPE_LANDMARKS["LEFT_WRIST"]
    RIGHT_WRIST = MEDIAPIPE_LANDMARKS["RIGHT_WRIST"]
    
    def __init__(
        self,
        config: Optional[TransformConfig] = None,
        action_spec: Optional[ActionSpec] = None,
    ):
        self.config = config or TransformConfig()
        self.action_spec = action_spec or ActionSpec()
        self.eps = 1e-8
        
    def compute_action(
        self,
        pose: np.ndarray,
        rotation: Optional[np.ndarray] = None,
        gripper_state: Optional[np.ndarray] = None,
        dt: float = 0.0333,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Action 벡터 계산
        
        Args:
            pose: [T, J, 3] 포즈 시퀀스
            rotation: [T, 2, 4] EEF 쿼터니언 (선택)
            gripper_state: [T, 2] Gripper 상태 (선택)
            dt: 시간 간격
            
        Returns:
            actions: [T-1, A] Action 벡터
            masks: [T-1] 유효 마스크
        """
        T = pose.shape[0]
        action_parts = []
        
        # 1. Position delta
        if self.action_spec.position_delta:
            if self.action_spec.eef_only:
                # EEF만 (양손 손목)
                eef_pos = np.stack([
                    pose[:, self.LEFT_WRIST],
                    pose[:, self.RIGHT_WRIST],
                ], axis=1)  # [T, 2, 3]
                delta_pos = eef_pos[1:] - eef_pos[:-1]  # [T-1, 2, 3]
            else:
                # 전체 관절
                delta_pos = pose[1:] - pose[:-1]  # [T-1, J, 3]
                
            delta_pos_flat = delta_pos.reshape(T - 1, -1)
            action_parts.append(delta_pos_flat)
            
        # 2. Rotation delta
        if self.action_spec.rotation_delta and rotation is not None:
            delta_rot = self._compute_rotation_delta(rotation)
            
            if self.action_spec.rotation_repr == "quaternion":
                delta_rot_flat = delta_rot.reshape(T - 1, -1)
            else:
                # Axis-angle 변환
                delta_rot_aa = self._quaternion_to_axis_angle(delta_rot)
                delta_rot_flat = delta_rot_aa.reshape(T - 1, -1)
                
            action_parts.append(delta_rot_flat)
            
        # 3. Gripper state
        if self.action_spec.gripper_state and gripper_state is not None:
            # t+1 시점의 gripper 상태 사용
            gripper = gripper_state[1:]  # [T-1, 2]
            action_parts.append(gripper)
            
        # Action 벡터 결합
        if action_parts:
            actions = np.concatenate(action_parts, axis=1)
        else:
            # 기본: EEF position delta
            eef_pos = np.stack([
                pose[:, self.LEFT_WRIST],
                pose[:, self.RIGHT_WRIST],
            ], axis=1)
            actions = (eef_pos[1:] - eef_pos[:-1]).reshape(T - 1, -1)
            
        masks = np.ones(T - 1, dtype=bool)
        
        logger.debug(f"Computed actions: shape={actions.shape}, spec={self.action_spec.to_string()}")
        
        return actions, masks
    
    def _compute_rotation_delta(
        self,
        rotations: np.ndarray,
    ) -> np.ndarray:
        """
        쿼터니언 차분 계산: Δq = q_{t+1} ⊗ q_t^{-1}
        
        Args:
            rotations: [T, N, 4] 쿼터니언 (w, x, y, z)
            
        Returns:
            [T-1, N, 4] 회전 차분
        """
        T, N, _ = rotations.shape
        delta = np.zeros((T - 1, N, 4))
        
        for t in range(T - 1):
            for n in range(N):
                q_t = rotations[t, n]
                q_t1 = rotations[t + 1, n]
                
                # q_t의 역 (conjugate for unit quaternion)
                q_t_inv = np.array([q_t[0], -q_t[1], -q_t[2], -q_t[3]])
                
                # 쿼터니언 곱셈
                delta[t, n] = self._quaternion_multiply(q_t1, q_t_inv)
                
        return delta
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """쿼터니언 곱셈"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def _quaternion_to_axis_angle(self, quaternions: np.ndarray) -> np.ndarray:
        """쿼터니언을 axis-angle로 변환"""
        T, N, _ = quaternions.shape
        axis_angles = np.zeros((T, N, 3))
        
        for t in range(T):
            for n in range(N):
                q = quaternions[t, n]
                rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy는 (x,y,z,w)
                axis_angles[t, n] = rot.as_rotvec()
                
        return axis_angles
    
    def estimate_gripper_state(
        self,
        hand_landmarks: np.ndarray,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """
        손가락 랜드마크에서 그리퍼 상태 추정
        
        Args:
            hand_landmarks: [T, 2, 21, 3] 양손 랜드마크
            threshold: Pinch 판정 임계값
            
        Returns:
            [T, 2] Gripper 상태 (0: open, 1: closed)
        """
        T = hand_landmarks.shape[0]
        gripper = np.zeros((T, 2))
        
        # MediaPipe Hand 인덱스
        THUMB_TIP = 4
        INDEX_TIP = 8
        
        for t in range(T):
            for hand in range(2):
                thumb = hand_landmarks[t, hand, THUMB_TIP]
                index = hand_landmarks[t, hand, INDEX_TIP]
                distance = np.linalg.norm(thumb - index)
                
                gripper[t, hand] = 1.0 if distance < threshold else 0.0
                
        return gripper
    
    def estimate_eef_rotation(
        self,
        pose: np.ndarray,
        hand_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        EEF 회전 추정 (손목/팔 방향 기반)
        
        Args:
            pose: [T, J, 3] 포즈
            hand_landmarks: [T, 2, 21, 3] 손 랜드마크 (선택)
            
        Returns:
            [T, 2, 4] EEF 쿼터니언
        """
        T = pose.shape[0]
        rotations = np.zeros((T, 2, 4))
        rotations[:, :, 0] = 1.0  # Identity quaternion (w=1)
        
        LEFT_ELBOW = MEDIAPIPE_LANDMARKS["LEFT_ELBOW"]
        RIGHT_ELBOW = MEDIAPIPE_LANDMARKS["RIGHT_ELBOW"]
        
        for t in range(T):
            # 왼손
            forearm_left = pose[t, self.LEFT_WRIST] - pose[t, LEFT_ELBOW]
            if np.linalg.norm(forearm_left) > self.eps:
                forearm_left = forearm_left / np.linalg.norm(forearm_left)
                rot = self._direction_to_quaternion(forearm_left)
                rotations[t, 0] = rot
                
            # 오른손
            forearm_right = pose[t, self.RIGHT_WRIST] - pose[t, RIGHT_ELBOW]
            if np.linalg.norm(forearm_right) > self.eps:
                forearm_right = forearm_right / np.linalg.norm(forearm_right)
                rot = self._direction_to_quaternion(forearm_right)
                rotations[t, 1] = rot
                
        return rotations
    
    def _direction_to_quaternion(self, direction: np.ndarray) -> np.ndarray:
        """방향 벡터를 쿼터니언으로 변환"""
        # Z축을 해당 방향으로 정렬
        z = direction
        
        # X축: 임의의 수직 벡터
        up = np.array([0, 1, 0])
        if np.abs(np.dot(z, up)) > 0.99:
            up = np.array([1, 0, 0])
        x = np.cross(up, z)
        x = x / (np.linalg.norm(x) + self.eps)
        
        # Y축
        y = np.cross(z, x)
        
        # 회전 행렬에서 쿼터니언
        rot_matrix = np.array([x, y, z]).T
        rot = Rotation.from_matrix(rot_matrix)
        q = rot.as_quat()  # (x, y, z, w)
        
        return np.array([q[3], q[0], q[1], q[2]])  # (w, x, y, z)


class StateActionEncoder:
    """
    State-Action 인코딩 통합 파이프라인
    """
    
    def __init__(
        self,
        config: Optional[TransformConfig] = None,
        state_spec: Optional[StateSpec] = None,
        action_spec: Optional[ActionSpec] = None,
    ):
        self.config = config or TransformConfig()
        self.state_spec = state_spec or StateSpec()
        self.action_spec = action_spec or ActionSpec()
        
        self.state_builder = StateBuilder(config, state_spec)
        self.action_computer = ActionComputer(config, action_spec)
        
    def encode(
        self,
        pose: np.ndarray,
        velocity: np.ndarray,
        timestamps: np.ndarray,
        obj_rel: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
        hand_landmarks: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateActionPair:
        """
        State-Action 쌍 생성
        
        Args:
            pose: [T, J, 3] 정규화된 포즈
            velocity: [T, J, 3] 속도
            timestamps: [T] 타임스탬프
            obj_rel: [T, K, 3] 객체 상대 위치
            conf: [T, J] 신뢰도
            hand_landmarks: [T, 2, 21, 3] 손 랜드마크
            metadata: 추가 메타데이터
            
        Returns:
            StateActionPair: 인코딩된 State-Action 쌍
        """
        T = pose.shape[0]
        dt = timestamps[1] - timestamps[0] if T > 1 else 0.0333
        
        # 1. State 생성
        states, state_masks = self.state_builder.build_state(
            pose=pose,
            velocity=velocity,
            obj_rel=obj_rel,
            conf=conf,
        )
        
        # 2. Gripper 상태 추정 (손 랜드마크가 있으면)
        gripper_state = None
        if hand_landmarks is not None and self.action_spec.gripper_state:
            gripper_state = self.action_computer.estimate_gripper_state(hand_landmarks)
            
        # 3. EEF 회전 추정
        eef_rotation = None
        if self.action_spec.rotation_delta:
            eef_rotation = self.action_computer.estimate_eef_rotation(pose, hand_landmarks)
            
        # 4. Action 생성
        actions, action_masks = self.action_computer.compute_action(
            pose=pose,
            rotation=eef_rotation,
            gripper_state=gripper_state,
            dt=dt,
        )
        
        # 5. 마스크 결합
        masks = state_masks.copy()
        
        # 6. 메타데이터 구성
        if metadata is None:
            metadata = {}
        metadata.update({
            "dt": dt,
            "fps": 1.0 / dt if dt > 0 else 30,
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
            "state_spec": self.state_spec.to_string(),
            "action_spec": self.action_spec.to_string(),
            "transform_version": self.config.version,
        })
        
        logger.info(
            f"Encoded State-Action: states={states.shape}, actions={actions.shape}, "
            f"dt={dt:.4f}s"
        )
        
        return StateActionPair(
            states=states,
            actions=actions,
            timestamps=timestamps,
            masks=masks,
            state_spec=self.state_spec,
            action_spec=self.action_spec,
            metadata=metadata,
        )
    
    def validate(self, pair: StateActionPair) -> Dict[str, Any]:
        """
        State-Action 쌍 검증
        
        Returns:
            검증 결과 딕셔너리
        """
        results = {
            "valid": True,
            "issues": [],
        }
        
        T_states = pair.states.shape[0]
        T_actions = pair.actions.shape[0]
        T_timestamps = len(pair.timestamps)
        
        # Shape 검증
        if T_timestamps != T_states:
            results["valid"] = False
            results["issues"].append(
                f"timestamps({T_timestamps}) != states({T_states})"
            )
            
        if T_actions != T_states - 1:
            results["valid"] = False
            results["issues"].append(
                f"actions({T_actions}) != states-1({T_states - 1})"
            )
            
        # dt 일관성 검증
        if T_timestamps > 1:
            dts = np.diff(pair.timestamps)
            dt_mean = dts.mean()
            dt_std = dts.std()
            if dt_std > dt_mean * 0.1:  # 10% 이상 편차
                results["issues"].append(
                    f"Inconsistent dt: mean={dt_mean:.4f}, std={dt_std:.4f}"
                )
                
        # NaN 검사
        if np.any(np.isnan(pair.states)):
            results["valid"] = False
            results["issues"].append("NaN in states")
            
        if np.any(np.isnan(pair.actions)):
            results["valid"] = False
            results["issues"].append("NaN in actions")
            
        # 범위 검사
        state_max = np.abs(pair.states).max()
        if state_max > 100:
            results["issues"].append(f"Large state values: max={state_max:.2f}")
            
        action_max = np.abs(pair.actions).max()
        if action_max > 10:
            results["issues"].append(f"Large action values: max={action_max:.2f}")
            
        results["stats"] = {
            "T": T_states,
            "state_dim": pair.states.shape[1],
            "action_dim": pair.actions.shape[1],
            "state_range": [pair.states.min(), pair.states.max()],
            "action_range": [pair.actions.min(), pair.actions.max()],
        }
        
        return results
