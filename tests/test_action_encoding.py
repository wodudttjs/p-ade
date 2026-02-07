"""
Action Encoding 테스트

MVP Phase 2 Week 7: Action Encoding
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformation.encoding import StateBuilder, ActionComputer, StateActionPair
from transformation.spec import StateSpec, ActionSpec, TransformConfig


class TestStateBuilder:
    """StateBuilder 테스트"""
    
    @pytest.fixture
    def state_builder(self):
        return StateBuilder()
    
    @pytest.fixture
    def sample_poses(self):
        """샘플 포즈 데이터 [T, J, 3]"""
        np.random.seed(42)
        T, J = 100, 33
        poses = np.random.randn(T, J, 3).astype(np.float32) * 0.1
        # visibility 추가 -> [T, J, 4]
        visibility = np.ones((T, J, 1))
        return np.concatenate([poses, visibility], axis=2)
    
    def test_build_state_basic(self, state_builder, sample_poses):
        """기본 상태 생성"""
        states, masks = state_builder.build_state(sample_poses)
        
        assert states.shape[0] == 100  # T
        assert masks.shape[0] == 100
        assert all(masks)  # 모두 유효
    
    def test_build_state_with_velocity(self, state_builder, sample_poses):
        """속도 포함 상태 생성"""
        velocity = np.zeros_like(sample_poses[:, :, :3])
        
        states, masks = state_builder.build_state(
            sample_poses,
            velocity=velocity,
        )
        
        assert states.shape[0] == 100
    
    def test_state_spec_customization(self, sample_poses):
        """StateSpec 커스터마이징"""
        spec = StateSpec(
            joint_positions=True,
            joint_velocities=False,
            confidence_stats=False,
        )
        builder = StateBuilder(state_spec=spec)
        
        states, _ = builder.build_state(sample_poses)
        
        # 위치만 -> 33 * 4 = 132
        assert states.shape[1] == 33 * 4


class TestActionComputer:
    """ActionComputer 테스트"""
    
    @pytest.fixture
    def action_computer(self):
        return ActionComputer()
    
    @pytest.fixture
    def sample_poses(self):
        """샘플 포즈 [T, J, 3]"""
        np.random.seed(42)
        T, J = 100, 33
        
        # 부드러운 움직임
        t = np.linspace(0, 2*np.pi, T)
        poses = np.zeros((T, J, 3))
        
        for j in range(J):
            poses[:, j, 0] = np.sin(t + j * 0.1) * 0.1
            poses[:, j, 1] = np.cos(t + j * 0.1) * 0.1
            poses[:, j, 2] = 0
        
        return poses.astype(np.float32)
    
    def test_compute_action_basic(self, action_computer, sample_poses):
        """기본 액션 계산"""
        actions, masks = action_computer.compute_action(sample_poses)
        
        assert actions.shape[0] == 99  # T-1
        assert masks.shape[0] == 99
        assert all(masks)
    
    def test_action_shape_eef_only(self, action_computer, sample_poses):
        """EEF만 사용 시 shape"""
        action_computer.action_spec.eef_only = True
        actions, _ = action_computer.compute_action(sample_poses)
        
        # 양손 * 3D = 6
        assert actions.shape[1] == 6
    
    def test_action_shape_all_joints(self, sample_poses):
        """전체 관절 사용 시 shape"""
        spec = ActionSpec(eef_only=False, position_delta=True)
        computer = ActionComputer(action_spec=spec)
        
        actions, _ = computer.compute_action(sample_poses)
        
        # 33 joints * 3D = 99
        assert actions.shape[1] == 99
    
    def test_action_delta_calculation(self, action_computer):
        """Delta 계산 정확성"""
        # 단순 선형 이동
        T, J = 10, 33
        poses = np.zeros((T, J, 3))
        poses[:, 15, 0] = np.arange(T) * 0.1  # LEFT_WRIST x 방향 이동
        poses[:, 16, 0] = np.arange(T) * 0.1  # RIGHT_WRIST x 방향 이동
        
        actions, _ = action_computer.compute_action(poses.astype(np.float32))
        
        # 각 step에서 delta = 0.1
        expected_delta = 0.1
        np.testing.assert_array_almost_equal(
            actions[:, 0],  # left wrist x
            np.full(T-1, expected_delta),
            decimal=5
        )
    
    def test_estimate_gripper_state(self, action_computer):
        """Gripper 상태 추정"""
        T = 50
        # [T, 2, 21, 3] 손 랜드마크
        hands = np.random.rand(T, 2, 21, 3).astype(np.float32)
        
        # Thumb tip (4)과 Index tip (8)을 가깝게
        hands[:, 0, 4, :] = hands[:, 0, 8, :] + 0.01  # 왼손 pinch
        
        gripper = action_computer.estimate_gripper_state(hands, threshold=0.05)
        
        assert gripper.shape == (T, 2)
        # 왼손은 closed (1.0)
        assert np.all(gripper[:, 0] == 1.0)


class TestActionEncoderIntegration:
    """ActionEncoder 통합 테스트"""
    
    @pytest.fixture
    def temp_poses_dir(self, tmp_path):
        """임시 포즈 디렉토리"""
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        
        # 테스트 포즈 파일 생성
        for i in range(3):
            T = 100 + i * 50
            poses = np.random.rand(T, 33, 3).astype(np.float32)
            confidence = np.random.rand(T).astype(np.float32)
            
            np.savez_compressed(
                poses_dir / f"test_{i}_pose.npz",
                body=poses,
                confidence=confidence,
                fps=30.0,
            )
        
        return poses_dir
    
    def test_encoder_imports(self):
        """Encoder 모듈 임포트"""
        from encode_actions import ActionEncoder, EncodingResult
        assert ActionEncoder is not None
        assert EncodingResult is not None
    
    def test_encode_single_file(self, temp_poses_dir, tmp_path):
        """단일 파일 인코딩"""
        from encode_actions import ActionEncoder
        
        output_dir = tmp_path / "episodes"
        encoder = ActionEncoder(
            poses_dir=str(temp_poses_dir),
            output_dir=str(output_dir),
        )
        
        file_path = temp_poses_dir / "test_0_pose.npz"
        result = encoder.encode_file(file_path)
        
        assert result.success
        assert result.num_frames == 100
        assert result.state_dim > 0
        assert result.action_dim > 0
        assert Path(result.output_path).exists()
    
    def test_encode_all_files(self, temp_poses_dir, tmp_path):
        """모든 파일 인코딩"""
        from encode_actions import ActionEncoder
        
        output_dir = tmp_path / "episodes"
        encoder = ActionEncoder(
            poses_dir=str(temp_poses_dir),
            output_dir=str(output_dir),
        )
        
        results = encoder.encode_all()
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # 출력 파일 확인
        output_files = list(output_dir.glob("*_episode.npz"))
        assert len(output_files) == 3
    
    def test_output_file_structure(self, temp_poses_dir, tmp_path):
        """출력 파일 구조 확인"""
        from encode_actions import ActionEncoder
        
        output_dir = tmp_path / "episodes"
        encoder = ActionEncoder(
            poses_dir=str(temp_poses_dir),
            output_dir=str(output_dir),
        )
        
        result = encoder.encode_file(temp_poses_dir / "test_0_pose.npz")
        
        # 출력 파일 로드
        data = np.load(result.output_path)
        
        # 필수 키 확인
        required_keys = [
            "states", "actions", "state_masks", "action_masks",
            "poses", "velocity", "timestamps",
            "fps", "video_id", "state_dim", "action_dim", "num_frames",
        ]
        
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Shape 확인
        T = int(data["num_frames"])
        assert data["states"].shape[0] == T
        assert data["actions"].shape[0] == T - 1
    
    def test_velocity_calculation(self, temp_poses_dir, tmp_path):
        """속도 계산 검증"""
        from encode_actions import ActionEncoder
        
        encoder = ActionEncoder(
            poses_dir=str(temp_poses_dir),
            output_dir=str(tmp_path / "episodes"),
        )
        
        # 선형 이동하는 포즈
        poses = np.zeros((100, 33, 3))
        poses[:, 0, 0] = np.arange(100) * 0.1  # x 방향 0.1씩 이동
        
        velocity = encoder.compute_velocity(poses, fps=30.0)
        
        # 속도 = 0.1 * 30 = 3.0
        expected = 0.1 * 30.0
        np.testing.assert_array_almost_equal(
            velocity[1:-1, 0, 0],
            np.full(98, expected),
            decimal=3
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
