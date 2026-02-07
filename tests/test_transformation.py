"""
MODULE 4: Data Transformation 테스트

FR-4.1 ~ FR-4.4 전체 테스트
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from transformation.spec import (
    TransformConfig,
    StateSpec,
    ActionSpec,
    MEDIAPIPE_LANDMARKS,
    LEFT_RIGHT_PAIRS,
    get_joint_indices,
    get_mirror_swap_indices,
    create_default_config,
)
from transformation.normalization import (
    ReferenceFrameAligner,
    RelativeCoordinateComputer,
    DimensionalityReducer,
    CoordinateNormalizer,
    FrameTransform,
)
from transformation.temporal import (
    TemporalResampler,
    SavgolSmoother,
    TemporalAligner,
)
from transformation.encoding import (
    StateBuilder,
    ActionComputer,
    StateActionEncoder,
)
from transformation.augment import (
    SpatialAugmenter,
    TemporalAugmenter,
    ViewpointAugmenter,
    DataAugmentationPipeline,
)
from transformation.export import (
    NpzExporter,
    ParquetExporter,
    HDF5Exporter,
    FormatConverter,
)


# ============== Fixtures ==============

@pytest.fixture
def sample_pose():
    """샘플 3D 포즈 데이터 생성 (T=100, J=33, 3)"""
    np.random.seed(42)
    T, J = 100, 33
    
    # 기본 포즈 구조 생성 (사람 형태 모방)
    pose = np.random.randn(T, J, 3) * 0.1
    
    # 핵심 관절 위치 설정
    # 골반 (LEFT_HIP, RIGHT_HIP)
    pose[:, 23, :] = [0.1, 0, 0]  # LEFT_HIP
    pose[:, 24, :] = [-0.1, 0, 0]  # RIGHT_HIP
    
    # 어깨 (LEFT_SHOULDER, RIGHT_SHOULDER)
    pose[:, 11, :] = [0.15, 0.5, 0]  # LEFT_SHOULDER
    pose[:, 12, :] = [-0.15, 0.5, 0]  # RIGHT_SHOULDER
    
    # 손목
    pose[:, 15, :] = [0.3, 0.3, 0.1]  # LEFT_WRIST
    pose[:, 16, :] = [-0.3, 0.3, 0.1]  # RIGHT_WRIST
    
    # 시간에 따른 약간의 움직임 추가
    for t in range(T):
        pose[t, 15, 0] += 0.05 * np.sin(t * 0.1)  # 왼손 움직임
        pose[t, 16, 0] += 0.05 * np.cos(t * 0.1)  # 오른손 움직임
        
    return pose.astype(np.float32)


@pytest.fixture
def sample_conf():
    """샘플 신뢰도 데이터"""
    np.random.seed(42)
    T, J = 100, 33
    conf = np.random.uniform(0.6, 1.0, (T, J)).astype(np.float32)
    return conf


@pytest.fixture
def sample_timestamps():
    """샘플 타임스탬프 (불균일 간격)"""
    T = 100
    # 약간 불균일한 타임스탬프
    base = np.arange(T) / 30.0  # 30fps 기준
    noise = np.random.uniform(-0.005, 0.005, T)
    noise[0] = 0  # 첫 타임스탬프는 0
    return (base + noise).astype(np.float32)


@pytest.fixture
def config():
    """기본 설정"""
    return TransformConfig()


# ============== FR-4.1 Coordinate Normalization 테스트 ==============

class TestSpec:
    """Spec 모듈 테스트"""
    
    def test_mediapipe_landmarks(self):
        """MediaPipe 랜드마크 정의 확인"""
        assert len(MEDIAPIPE_LANDMARKS) == 33
        assert MEDIAPIPE_LANDMARKS["LEFT_WRIST"] == 15
        assert MEDIAPIPE_LANDMARKS["RIGHT_WRIST"] == 16
        
    def test_left_right_pairs(self):
        """좌우 대칭 쌍 확인"""
        assert len(LEFT_RIGHT_PAIRS) == 16
        for left, right in LEFT_RIGHT_PAIRS:
            assert "LEFT" in left
            assert "RIGHT" in right
            
    def test_get_joint_indices(self):
        """관절 인덱스 추출"""
        names = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_ELBOW"]
        indices = get_joint_indices(names)
        assert indices == [15, 16, 13]
        
    def test_get_mirror_swap_indices(self):
        """미러 스왑 인덱스"""
        names = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        swap = get_mirror_swap_indices(names)
        assert 0 in swap and swap[0] == 1  # LEFT_WRIST ↔ RIGHT_WRIST
        assert 2 in swap and swap[2] == 3  # LEFT_SHOULDER ↔ RIGHT_SHOULDER
        
    def test_transform_config(self):
        """설정 객체"""
        config = create_default_config()
        assert config.target_fps == 30
        assert config.conf_min == 0.5
        assert config.version == "4.0.0"
        
    def test_config_to_dict(self, config):
        """설정 딕셔너리 변환"""
        d = config.to_dict()
        assert "target_fps" in d
        assert "normalization" in d
        assert "augment" in d
        
    def test_config_params_hash(self, config):
        """설정 해시 생성"""
        hash1 = config.get_params_hash()
        config.target_fps = 60
        hash2 = config.get_params_hash()
        assert hash1 != hash2
        
    def test_state_spec(self):
        """State 스키마"""
        spec = StateSpec()
        dim = spec.get_state_dim()
        assert dim > 0
        schema = spec.to_string()
        assert "pos" in schema
        
    def test_action_spec(self):
        """Action 스키마"""
        spec = ActionSpec()
        dim = spec.get_action_dim()
        assert dim > 0
        schema = spec.to_string()
        assert "dpos" in schema


class TestReferenceFrameAligner:
    """Reference Frame Alignment 테스트"""
    
    def test_align_basic(self, sample_pose, sample_conf):
        """기본 정렬"""
        aligner = ReferenceFrameAligner()
        result = aligner.align(sample_pose, sample_conf)
        
        assert result.pose_aligned.shape == sample_pose.shape
        assert len(result.transforms) == len(sample_pose)
        assert result.valid_mask.shape == (len(sample_pose),)
        
    def test_align_pelvis_centered(self, sample_pose):
        """골반 중심 정렬 확인"""
        aligner = ReferenceFrameAligner()
        result = aligner.align(sample_pose)
        
        # 정렬 후 골반 중심이 원점 근처인지 확인
        LEFT_HIP = 23
        RIGHT_HIP = 24
        pelvis = (result.pose_aligned[:, LEFT_HIP] + result.pose_aligned[:, RIGHT_HIP]) / 2
        
        # 완벽히 0은 아닐 수 있지만 작은 값이어야 함
        assert np.abs(pelvis).mean() < 0.5
        
    def test_align_without_conf(self, sample_pose):
        """신뢰도 없이 정렬"""
        aligner = ReferenceFrameAligner()
        result = aligner.align(sample_pose, None)
        
        assert result.pose_aligned.shape == sample_pose.shape
        
    def test_quality_metrics(self, sample_pose, sample_conf):
        """품질 메트릭"""
        aligner = ReferenceFrameAligner()
        result = aligner.align(sample_pose, sample_conf)
        
        assert "valid_ratio" in result.quality_metrics
        assert "scale_mean" in result.quality_metrics


class TestRelativeCoordinateComputer:
    """Relative Coordinate 테스트"""
    
    def test_compute_joint_relative_parent(self, sample_pose):
        """부모 관절 기준 상대 좌표"""
        computer = RelativeCoordinateComputer()
        relative = computer.compute_joint_relative(sample_pose, method="parent")
        
        assert relative.shape == sample_pose.shape
        
    def test_compute_joint_relative_reference(self, sample_pose):
        """기준 관절 기준 상대 좌표"""
        computer = RelativeCoordinateComputer()
        relative = computer.compute_joint_relative(
            sample_pose, method="reference", reference_joint="LEFT_HIP"
        )
        
        assert relative.shape == sample_pose.shape
        
    def test_compute_eef_positions(self, sample_pose):
        """EEF 위치 추출"""
        computer = RelativeCoordinateComputer()
        eef = computer.compute_eef_positions(sample_pose)
        
        assert eef.shape == (len(sample_pose), 2, 3)


class TestDimensionalityReducer:
    """Dimensionality Reduction 테스트"""
    
    def test_select_joints(self, sample_pose, config):
        """관절 선택"""
        reducer = DimensionalityReducer(config)
        selected = reducer.select_joints(sample_pose)
        
        assert selected.shape[0] == sample_pose.shape[0]
        assert selected.shape[1] == len(config.joints_keep)
        assert selected.shape[2] == 3
        
    def test_compress_dtype(self, sample_pose):
        """dtype 압축"""
        reducer = DimensionalityReducer()
        
        compressed16 = reducer.compress_dtype(sample_pose, "float16")
        assert compressed16.dtype == np.float16
        
        compressed32 = reducer.compress_dtype(sample_pose, "float32")
        assert compressed32.dtype == np.float32


class TestCoordinateNormalizer:
    """통합 좌표 정규화 테스트"""
    
    def test_normalize(self, sample_pose, sample_conf):
        """전체 정규화 파이프라인"""
        normalizer = CoordinateNormalizer()
        result = normalizer.normalize(sample_pose, sample_conf)
        
        assert "pose_aligned" in result
        assert "pose_selected" in result
        assert "pose_compressed" in result
        assert "eef_positions" in result


# ============== FR-4.2 Action Encoding 테스트 ==============

class TestTemporalResampler:
    """Temporal Resampling 테스트"""
    
    def test_resample(self, sample_pose, sample_timestamps):
        """기본 리샘플링"""
        resampler = TemporalResampler()
        result = resampler.resample(sample_timestamps, sample_pose)
        
        assert len(result.timestamps) > 0
        assert result.data.shape[1:] == sample_pose.shape[1:]
        assert len(result.mask_valid) == len(result.timestamps)
        
    def test_resample_uniform(self, sample_pose):
        """균일 간격 확인"""
        timestamps = np.arange(len(sample_pose)) / 30.0
        resampler = TemporalResampler()
        result = resampler.resample(timestamps, sample_pose)
        
        if len(result.timestamps) > 1:
            dts = np.diff(result.timestamps)
            assert np.std(dts) < 0.001  # 균일 간격


class TestSavgolSmoother:
    """Savitzky-Golay 스무딩 테스트"""
    
    def test_smooth(self, sample_pose):
        """기본 스무딩"""
        smoother = SavgolSmoother()
        smoothed = smoother.smooth(sample_pose)
        
        assert smoothed.shape == sample_pose.shape
        
    def test_smooth_preserves_shape(self, sample_pose):
        """Shape 보존"""
        smoother = SavgolSmoother()
        smoothed = smoother.smooth(sample_pose, window=5, polyorder=2)
        
        assert smoothed.shape == sample_pose.shape


class TestTemporalAligner:
    """Temporal Alignment 테스트"""
    
    def test_align(self, sample_pose, sample_timestamps, sample_conf):
        """시간축 정렬"""
        aligner = TemporalAligner()
        result = aligner.align(sample_timestamps, sample_pose, sample_conf)
        
        assert "timestamps" in result
        assert "pose" in result
        assert "pose_smoothed" in result
        assert "dt" in result
        
    def test_compute_velocity(self, sample_pose):
        """속도 계산"""
        aligner = TemporalAligner()
        velocity = aligner.compute_velocity(sample_pose, dt=0.0333)
        
        assert velocity.shape == sample_pose.shape


class TestStateBuilder:
    """State Builder 테스트"""
    
    def test_build_state(self, sample_pose):
        """상태 벡터 생성"""
        builder = StateBuilder()
        velocity = np.zeros_like(sample_pose)
        states, masks = builder.build_state(sample_pose, velocity)
        
        assert states.ndim == 2
        assert states.shape[0] == sample_pose.shape[0]
        
    def test_build_state_with_objects(self, sample_pose):
        """객체 포함 상태"""
        builder = StateBuilder(state_spec=StateSpec(object_relations=True))
        obj_rel = np.random.randn(len(sample_pose), 3, 3).astype(np.float32)
        
        states, masks = builder.build_state(sample_pose, obj_rel=obj_rel)
        assert states.ndim == 2


class TestActionComputer:
    """Action Computer 테스트"""
    
    def test_compute_action(self, sample_pose):
        """액션 계산"""
        computer = ActionComputer()
        actions, masks = computer.compute_action(sample_pose)
        
        assert actions.shape[0] == sample_pose.shape[0] - 1
        
    def test_estimate_gripper_state(self):
        """그리퍼 상태 추정"""
        computer = ActionComputer()
        hand_landmarks = np.random.randn(50, 2, 21, 3).astype(np.float32)
        gripper = computer.estimate_gripper_state(hand_landmarks)
        
        assert gripper.shape == (50, 2)
        assert np.all((gripper == 0) | (gripper == 1))


class TestStateActionEncoder:
    """State-Action Encoder 테스트"""
    
    def test_encode(self, sample_pose, sample_timestamps):
        """인코딩"""
        encoder = StateActionEncoder()
        velocity = np.zeros_like(sample_pose)
        
        pair = encoder.encode(
            pose=sample_pose,
            velocity=velocity,
            timestamps=sample_timestamps,
        )
        
        assert pair.states.shape[0] == len(sample_pose)
        assert pair.actions.shape[0] == len(sample_pose) - 1
        
    def test_validate(self, sample_pose, sample_timestamps):
        """검증"""
        encoder = StateActionEncoder()
        velocity = np.zeros_like(sample_pose)
        
        pair = encoder.encode(sample_pose, velocity, sample_timestamps)
        result = encoder.validate(pair)
        
        assert result["valid"]
        assert "stats" in result


# ============== FR-4.3 Format Conversion 테스트 ==============

class TestNpzExporter:
    """NPZ Exporter 테스트"""
    
    def test_save_and_load(self, sample_pose, sample_timestamps):
        """저장 및 로드"""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = NpzExporter(tmpdir)
            
            states = sample_pose.reshape(len(sample_pose), -1)
            actions = states[1:] - states[:-1]
            
            path = exporter.save(
                states=states,
                actions=actions,
                timestamps=sample_timestamps,
                metadata={"video_id": "test", "episode_id": "ep_001"},
                video_id="test",
                episode_id="ep_001",
            )
            
            assert path.exists()
            
            loaded = exporter.load(path)
            assert "states" in loaded
            assert "actions" in loaded
            assert np.allclose(loaded["states"], states, atol=1e-3)


class TestParquetExporter:
    """Parquet Exporter 테스트"""
    
    def test_save_frame_level(self, sample_pose, sample_timestamps):
        """프레임 단위 저장"""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ParquetExporter(tmpdir)
            
            states = sample_pose.reshape(len(sample_pose), -1)
            actions = states[1:] - states[:-1]
            
            path = exporter.save_frame_level(
                states=states,
                actions=actions,
                timestamps=sample_timestamps,
                metadata={"video_id": "test", "episode_id": "ep_001"},
            )
            
            assert path.exists()


class TestHDF5Exporter:
    """HDF5 Exporter 테스트"""
    
    def test_save_and_load(self, sample_pose, sample_timestamps):
        """저장 및 로드"""
        pytest.importorskip("h5py")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = HDF5Exporter(tmpdir)
            
            states = sample_pose.reshape(len(sample_pose), -1)
            actions = states[1:] - states[:-1]
            
            episodes = [{
                "states": states,
                "actions": actions,
                "timestamps": sample_timestamps,
                "metadata": {"video_id": "test", "episode_id": "ep_001"},
            }]
            
            path = exporter.save(episodes, "test_dataset")
            assert path.exists()
            
            loaded = exporter.load(path)
            assert len(loaded) == 1
            assert "states" in loaded[0]


class TestFormatConverter:
    """Format Converter 테스트"""
    
    def test_convert_npz(self, sample_pose, sample_timestamps):
        """NPZ 변환"""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = FormatConverter(tmpdir)
            
            states = sample_pose.reshape(len(sample_pose), -1)
            actions = states[1:] - states[:-1]
            
            paths = converter.convert(
                states=states,
                actions=actions,
                timestamps=sample_timestamps,
                metadata={"video_id": "test", "episode_id": "ep_001"},
                formats=["npz"],
            )
            
            assert "npz" in paths
            assert paths["npz"].exists()


# ============== FR-4.4 Data Augmentation 테스트 ==============

class TestSpatialAugmenter:
    """Spatial Augmenter 테스트"""
    
    def test_mirror(self, sample_pose):
        """미러링"""
        augmenter = SpatialAugmenter()
        result = augmenter.mirror(sample_pose)
        
        assert result.pose.shape == sample_pose.shape
        assert result.aug_type == "mirror"
        
        # X축 반전 확인 (좌우 스왑 전)
        # 미러링은 X반전 + 좌우 스왑이므로 단순 비교 불가
        # 대신 X 좌표의 부호가 바뀌었는지 확인
        assert not np.allclose(result.pose[:, :, 0], sample_pose[:, :, 0])
        
    def test_scale(self, sample_pose):
        """스케일 변화"""
        augmenter = SpatialAugmenter()
        result = augmenter.scale(sample_pose, scale_range=(0.9, 1.1))
        
        assert result.pose.shape == sample_pose.shape
        assert result.aug_type == "scale"
        assert "scale_factor" in result.aug_params
        
    def test_add_noise(self, sample_pose):
        """노이즈 추가"""
        augmenter = SpatialAugmenter()
        result = augmenter.add_noise(sample_pose, sigma=0.01)
        
        assert result.pose.shape == sample_pose.shape
        assert result.aug_type == "noise"
        
        # 원본과 달라야 함
        assert not np.allclose(result.pose, sample_pose)


class TestTemporalAugmenter:
    """Temporal Augmenter 테스트"""
    
    def test_speed_change(self, sample_pose, sample_timestamps):
        """속도 변화"""
        augmenter = TemporalAugmenter()
        result = augmenter.speed_change(sample_pose, sample_timestamps)
        
        assert result.aug_type == "speed_change"
        assert "speed_factor" in result.aug_params
        
    def test_random_offset(self, sample_pose, sample_timestamps):
        """랜덤 오프셋"""
        augmenter = TemporalAugmenter()
        result = augmenter.random_offset(sample_pose, sample_timestamps)
        
        assert result.aug_type == "offset"
        assert len(result.pose) <= len(sample_pose)
        
    def test_random_crop(self, sample_pose, sample_timestamps):
        """랜덤 크롭"""
        augmenter = TemporalAugmenter()
        result = augmenter.random_crop(sample_pose, sample_timestamps)
        
        assert result.aug_type == "crop"
        assert len(result.pose) <= len(sample_pose)


class TestViewpointAugmenter:
    """Viewpoint Augmenter 테스트"""
    
    def test_rotate_viewpoint(self, sample_pose):
        """시점 회전"""
        augmenter = ViewpointAugmenter()
        result = augmenter.rotate_viewpoint(sample_pose)
        
        assert result.pose.shape == sample_pose.shape
        assert result.aug_type == "viewpoint_rotation"
        assert "yaw" in result.aug_params
        
    def test_normalize_depth(self, sample_pose):
        """Depth 정규화"""
        augmenter = ViewpointAugmenter()
        result = augmenter.normalize_depth(sample_pose, method="median")
        
        assert result.pose.shape == sample_pose.shape
        assert result.aug_type == "depth_normalize"


class TestDataAugmentationPipeline:
    """Data Augmentation Pipeline 테스트"""
    
    def test_augment(self, sample_pose, sample_timestamps):
        """전체 증강 파이프라인"""
        pipeline = DataAugmentationPipeline()
        results = pipeline.augment(
            pose=sample_pose,
            timestamps=sample_timestamps,
            source_episode_id="test_ep",
        )
        
        assert len(results) > 0
        for result in results:
            assert result.source_episode_id == "test_ep"
            
    def test_augment_disabled(self, sample_pose, sample_timestamps):
        """증강 비활성화"""
        from transformation.spec import AugmentConfig
        config = AugmentConfig(enable=False)
        pipeline = DataAugmentationPipeline(config)
        
        results = pipeline.augment(sample_pose, sample_timestamps)
        assert len(results) == 0


# ============== 통합 테스트 ==============

class TestIntegration:
    """통합 테스트"""
    
    def test_full_pipeline(self, sample_pose, sample_conf, sample_timestamps):
        """전체 파이프라인 테스트"""
        # 1. 좌표 정규화
        normalizer = CoordinateNormalizer()
        norm_result = normalizer.normalize(sample_pose, sample_conf)
        
        # 2. 시간축 정렬
        aligner = TemporalAligner()
        aligned = aligner.align(
            sample_timestamps,
            norm_result["pose_aligned"],
            sample_conf,
        )
        
        # 3. 속도 계산
        velocity = aligner.compute_velocity(aligned["pose_smoothed"], aligned["dt"])
        
        # 4. State-Action 인코딩
        encoder = StateActionEncoder()
        pair = encoder.encode(
            pose=aligned["pose_smoothed"],
            velocity=velocity,
            timestamps=aligned["timestamps"],
        )
        
        # 5. 검증
        validation = encoder.validate(pair)
        assert validation["valid"]
        
        # 6. 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = FormatConverter(tmpdir)
            paths = converter.convert(
                states=pair.states,
                actions=pair.actions,
                timestamps=pair.timestamps,
                metadata=pair.metadata,
                formats=["npz"],
            )
            assert paths["npz"].exists()
            
    def test_shape_consistency(self, sample_pose, sample_timestamps):
        """Shape 일관성 검증"""
        aligner = TemporalAligner()
        aligned = aligner.align(sample_timestamps, sample_pose)
        
        T = len(aligned["timestamps"])
        
        # states.shape[0] == timestamps.shape[0]
        encoder = StateActionEncoder()
        velocity = aligner.compute_velocity(aligned["pose_smoothed"], aligned["dt"])
        pair = encoder.encode(aligned["pose_smoothed"], velocity, aligned["timestamps"])
        
        assert pair.states.shape[0] == T
        assert pair.actions.shape[0] == T - 1
        assert len(pair.timestamps) == T
