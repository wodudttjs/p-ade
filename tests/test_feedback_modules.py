"""
테스트: Feedback 반영 모듈

- 데이터 계약 (schemas)
- Job Key/멱등성 (job_manager)
- Quality Metric (quality_metrics)
- 재현성 (reproducibility)
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import json


class TestDataContracts:
    """데이터 계약 스키마 테스트"""
    
    def test_raw_candidate_creation(self):
        """RawCandidate 생성 테스트"""
        from schemas.contracts import RawCandidate
        
        candidate = RawCandidate(
            platform="youtube",
            video_id="abc123",
            source_url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            tags=["test", "demo"],
        )
        
        assert candidate.platform == "youtube"
        assert candidate.video_id == "abc123"
        assert candidate.url_hash is not None
        assert len(candidate.url_hash) == 16
    
    def test_raw_candidate_to_jsonl(self):
        """RawCandidate JSONL 변환 테스트"""
        from schemas.contracts import RawCandidate
        
        candidate = RawCandidate(
            platform="youtube",
            video_id="xyz789",
            source_url="https://youtube.com/watch?v=xyz789",
        )
        
        jsonl = candidate.to_jsonl()
        assert '"platform": "youtube"' in jsonl
        assert '"video_id": "xyz789"' in jsonl
    
    def test_download_manifest(self):
        """DownloadManifest 테스트"""
        from schemas.contracts import DownloadManifest
        
        manifest = DownloadManifest(
            video_id="test123",
            platform="youtube",
            local_path="/data/videos/test123.mp4",
            duration_sec=120.5,
            resolution="1920x1080",
            fps=30.0,
            yt_dlp_format_id="22",
            checksum_sha256="abc123...",
        )
        
        assert manifest.status == "success"
        json_str = manifest.to_json()
        data = json.loads(json_str)
        assert data["duration_sec"] == 120.5
    
    def test_episode_id_generation(self):
        """Episode ID 생성 규칙 테스트"""
        from schemas.contracts import EpisodeResult
        
        episode_id = EpisodeResult.generate_episode_id("video123", 5)
        assert episode_id == "video123_ep005"
        
        episode_id2 = EpisodeResult.generate_episode_id("abc", 42)
        assert episode_id2 == "abc_ep042"
    
    def test_episode_result(self):
        """EpisodeResult 테스트"""
        from schemas.contracts import EpisodeResult
        
        episode = EpisodeResult(
            episode_id="test_ep001",
            video_id="test",
            platform="youtube",
            episode_index=1,
            start_frame=0,
            end_frame=100,
            num_frames=100,
            states_shape=[100, 33, 3],
            actions_shape=[99, 33, 3],
            quality_score=0.85,
            action_type="kinematic_delta",
        )
        
        assert episode.action_type == "kinematic_delta"
        assert episode.num_frames == 100
    
    def test_job_key_utilities(self):
        """Job Key 유틸리티 테스트"""
        from schemas.contracts import generate_job_key, generate_result_path, parse_episode_id
        
        # Job key 생성
        key = generate_job_key("youtube", "abc123", "v1.0.0")
        assert key == "youtube_abc123_v1.0.0"
        
        # 결과 경로 생성
        path = generate_result_path("data", "youtube", "abc123", "v1.0.0")
        assert "youtube" in path
        assert "abc123" in path
        
        # Episode ID 파싱
        video_id, idx = parse_episode_id("video123_ep042")
        assert video_id == "video123"
        assert idx == 42


class TestJobManager:
    """Job Manager 테스트"""
    
    def test_job_key_creation(self):
        """JobKey 생성 테스트"""
        from core.job_manager import JobKey
        
        key = JobKey(
            platform="youtube",
            video_id="test123",
            processing_version="abc123"
        )
        
        assert str(key) == "youtube_test123_abc123"
    
    def test_job_key_from_string(self):
        """문자열에서 JobKey 파싱 테스트"""
        from core.job_manager import JobKey
        
        key = JobKey.from_string("youtube_video123_v1.0")
        
        assert key.platform == "youtube"
        assert key.video_id == "video123"
        assert key.processing_version == "v1.0"
    
    def test_job_key_result_path(self):
        """결과 경로 생성 테스트"""
        from core.job_manager import JobKey
        
        key = JobKey("youtube", "vid123", "v2")
        path = key.get_result_path("data")
        
        assert "episodes" in str(path)
        assert "youtube" in str(path)
        assert "vid123" in str(path)
    
    def test_job_manager_create_job_key(self):
        """JobManager job key 생성 테스트"""
        from core.job_manager import JobManager
        
        manager = JobManager(processing_version="test_v1")
        key = manager.create_job_key("youtube", "abc123")
        
        assert key.platform == "youtube"
        assert key.video_id == "abc123"
        assert key.processing_version == "test_v1"
    
    def test_job_status_enum(self):
        """JobStatus enum 테스트"""
        from core.job_manager import JobStatus
        
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
    
    def test_failure_type_enum(self):
        """FailureType enum 테스트 (3종)"""
        from core.job_manager import FailureType
        
        assert FailureType.NETWORK.value == "network"  # 재시도
        assert FailureType.QUALITY.value == "quality"  # 스킵
        assert FailureType.SYSTEM.value == "system"    # 즉시 알림
    
    def test_job_result(self):
        """JobResult 테스트"""
        from core.job_manager import JobResult, JobStatus, JobStage
        
        result = JobResult(
            job_key="youtube_test_v1",
            status=JobStatus.COMPLETED,
            stage=JobStage.EXTRACT,
            result_path="/data/episodes/youtube/test/v1/",
        )
        
        assert result.status == JobStatus.COMPLETED
        
        result_dict = result.to_dict()
        assert result_dict["status"] == "completed"
        assert result_dict["stage"] == "extract"


class TestQualityMetrics:
    """Quality Metrics 테스트"""
    
    def test_quality_thresholds_defaults(self):
        """QualityThresholds 기본값 테스트"""
        from core.quality_metrics import QualityThresholds
        
        thresholds = QualityThresholds()
        
        assert thresholds.min_confidence == 0.5
        assert thresholds.max_jitter_score == 0.3
        assert thresholds.min_episode_frames == 30
        assert thresholds.profile == "default"
    
    def test_quality_thresholds_custom(self):
        """QualityThresholds 커스텀 테스트"""
        from core.quality_metrics import QualityThresholds
        
        thresholds = QualityThresholds(
            min_confidence=0.7,
            max_jitter_score=0.2,
            profile="strict"
        )
        
        assert thresholds.min_confidence == 0.7
        assert thresholds.profile == "strict"
    
    def test_pose_quality_metrics_calculation(self):
        """포즈 품질 메트릭 계산 테스트"""
        from core.quality_metrics import QualityMetricCalculator, QualityThresholds
        
        # 좋은 품질의 테스트 데이터
        T, J = 100, 33
        poses = np.random.randn(T, J, 4).astype(np.float32)
        poses[:, :, 3] = 0.8  # visibility
        
        calc = QualityMetricCalculator()
        metrics = calc.calculate_pose_quality(poses)
        
        assert metrics.total_frames == T
        assert metrics.confidence_mean > 0
        assert 0 <= metrics.quality_score <= 1.0
    
    def test_nan_detection(self):
        """NaN 감지 테스트"""
        from core.quality_metrics import QualityMetricCalculator
        
        T, J = 50, 33
        poses = np.random.randn(T, J, 4).astype(np.float32)
        
        # 일부 프레임에 NaN 삽입
        poses[10:15, :, :] = np.nan
        
        calc = QualityMetricCalculator()
        metrics = calc.calculate_pose_quality(poses)
        
        assert metrics.nan_frames == 5
        assert metrics.nan_ratio == 0.1
    
    def test_jitter_calculation(self):
        """Jitter 계산 테스트"""
        from core.quality_metrics import QualityMetricCalculator
        
        T, J = 100, 33
        
        # 부드러운 움직임 (낮은 jitter)
        smooth_poses = np.zeros((T, J, 4))
        for t in range(T):
            smooth_poses[t, :, 0] = t / 100  # 선형 움직임
            smooth_poses[t, :, 3] = 0.9
        
        calc = QualityMetricCalculator()
        smooth_metrics = calc.calculate_pose_quality(smooth_poses)
        
        # 급격한 움직임 (높은 jitter)
        jerky_poses = np.random.randn(T, J, 4).astype(np.float32)
        jerky_poses[:, :, 3] = 0.9
        
        jerky_metrics = calc.calculate_pose_quality(jerky_poses)
        
        # 부드러운 움직임이 더 낮은 jitter를 가져야 함
        assert smooth_metrics.jitter_score < jerky_metrics.jitter_score
    
    def test_quality_distribution_tracker(self):
        """품질 분포 추적 테스트"""
        from core.quality_metrics import (
            QualityMetricCalculator, 
            QualityDistributionTracker,
            PoseQualityMetrics
        )
        
        tracker = QualityDistributionTracker()
        
        # 샘플 추가
        for i in range(10):
            metrics = PoseQualityMetrics(
                confidence_mean=0.7 + i * 0.02,
                jitter_score=0.1 + i * 0.01,
                nan_ratio=0.01,
                total_frames=100,
                quality_score=0.8,
                passed=True,
            )
            tracker.add_sample(metrics, category="walking")
        
        summary = tracker.get_distribution_summary()
        
        assert summary["total_samples"] == 10
        assert "confidence" in summary
        assert "jitter" in summary
        assert "category_balance" in summary


class TestReproducibility:
    """재현성 테스트"""
    
    def test_version_collector_python(self):
        """Python 버전 수집 테스트"""
        from core.reproducibility import VersionCollector
        
        version = VersionCollector.get_python_version()
        assert version  # 비어있지 않음
        assert "." in version  # "3.x.y" 형식
    
    def test_version_collector_package(self):
        """패키지 버전 수집 테스트"""
        from core.reproducibility import VersionCollector
        
        numpy_version = VersionCollector.get_package_version("numpy")
        assert numpy_version != "unknown"
    
    def test_reproducibility_context(self):
        """ReproducibilityContext 테스트"""
        from core.reproducibility import ReproducibilityContext
        
        context = ReproducibilityContext(
            git_commit_hash="abc123",
            python_version="3.10.0",
            mediapipe_version="0.10.0",
            processing_params={"fps": 30},
        )
        
        assert context.git_commit_hash == "abc123"
        
        fingerprint = context.get_fingerprint()
        assert len(fingerprint) == 12
    
    def test_version_collector_all(self):
        """전체 버전 수집 테스트"""
        from core.reproducibility import VersionCollector
        
        context = VersionCollector.collect_all(
            processing_params={"test": True}
        )
        
        assert context.python_version
        assert context.captured_at
        assert context.processing_params["test"] is True
    
    def test_reproducibility_manager(self):
        """ReproducibilityManager 테스트"""
        from core.reproducibility import ReproducibilityManager, ReproducibilityContext
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(base_dir=tmpdir)
            
            context = ReproducibilityContext(
                git_commit_hash="test123",
                python_version="3.10.0",
            )
            
            # 저장
            path = manager.save_context(context, "youtube", "video123")
            assert path.exists()
            
            # 로드
            loaded = manager.load_context("youtube", "video123")
            assert loaded is not None
            assert loaded.git_commit_hash == "test123"
    
    def test_context_comparison(self):
        """컨텍스트 비교 테스트"""
        from core.reproducibility import ReproducibilityManager, ReproducibilityContext
        
        ctx1 = ReproducibilityContext(
            git_commit_hash="v1",
            mediapipe_version="0.10.0",
        )
        
        ctx2 = ReproducibilityContext(
            git_commit_hash="v2",
            mediapipe_version="0.10.1",
        )
        
        manager = ReproducibilityManager()
        diffs = manager.compare_contexts(ctx1, ctx2)
        
        assert "git_commit_hash" in diffs
        assert "mediapipe_version" in diffs


class TestDatabaseModels:
    """데이터베이스 모델 테스트"""
    
    def test_processing_job_model(self):
        """ProcessingJob 모델 테스트"""
        from models.database import ProcessingJob
        
        # 클래스 메서드 테스트
        job_key = ProcessingJob.generate_job_key("youtube", "abc123", "v1.0")
        assert job_key == "youtube_abc123_v1.0"
        
        result_path = ProcessingJob.generate_result_path("youtube", "abc123", "v1.0")
        assert "episodes" in result_path
    
    def test_quality_config_model(self):
        """QualityConfig 모델 테스트"""
        from models.database import QualityConfig
        
        # 모델 필드 확인
        assert hasattr(QualityConfig, "min_confidence")
        assert hasattr(QualityConfig, "max_jitter_score")
        assert hasattr(QualityConfig, "profile")
        assert hasattr(QualityConfig, "expected_pass_rate")
    
    def test_video_license_fields(self):
        """Video 라이선스 필드 테스트"""
        from models.database import Video
        
        # 라이선스 관련 필드 확인
        assert hasattr(Video, "license")
        assert hasattr(Video, "copyright_owner")
        assert hasattr(Video, "permission_proof")
        assert hasattr(Video, "attribution_required")
        assert hasattr(Video, "is_download_allowed")
        assert hasattr(Video, "source_terms_snapshot")
    
    def test_episode_reproducibility_fields(self):
        """Episode 재현성 필드 테스트"""
        from models.database import Episode
        
        # 재현성 관련 필드 확인
        assert hasattr(Episode, "processing_version")
        assert hasattr(Episode, "model_versions")
        assert hasattr(Episode, "processing_params")
        assert hasattr(Episode, "job_key")


class TestDefaultProcessingParams:
    """기본 처리 파라미터 테스트"""
    
    def test_default_params_exist(self):
        """기본 파라미터 존재 테스트"""
        from core.reproducibility import DEFAULT_PROCESSING_PARAMS
        
        assert "sampling_fps" in DEFAULT_PROCESSING_PARAMS
        assert "pose_model_complexity" in DEFAULT_PROCESSING_PARAMS
        assert "min_detection_confidence" in DEFAULT_PROCESSING_PARAMS
        assert "conf_threshold" in DEFAULT_PROCESSING_PARAMS
    
    def test_get_current_context(self):
        """현재 컨텍스트 가져오기 테스트"""
        from core.reproducibility import get_current_reproducibility_context
        
        context = get_current_reproducibility_context()
        
        assert context.processing_params["sampling_fps"] == 30
        assert "min_detection_confidence" in context.processing_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
