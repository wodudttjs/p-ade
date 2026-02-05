"""
Quality Filtering 테스트

MVP Phase 2 Week 6: Quality Filtering
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quality_metrics import (
    QualityMetricCalculator,
    QualityThresholds,
    PoseQualityMetrics,
)


class TestQualityThresholds:
    """QualityThresholds 테스트"""
    
    def test_default_thresholds(self):
        """기본 임계값 테스트"""
        thresholds = QualityThresholds()
        
        assert thresholds.min_confidence == 0.5
        assert thresholds.max_jitter_score == 0.3
        assert thresholds.min_episode_frames == 30
        assert thresholds.max_nan_ratio == 0.1
        assert thresholds.profile == "default"
    
    def test_custom_thresholds(self):
        """커스텀 임계값 테스트"""
        thresholds = QualityThresholds(
            min_confidence=0.7,
            max_jitter_score=0.2,
            profile="strict",
        )
        
        assert thresholds.min_confidence == 0.7
        assert thresholds.max_jitter_score == 0.2
        assert thresholds.profile == "strict"
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        thresholds = QualityThresholds()
        d = thresholds.to_dict()
        
        assert "min_confidence" in d
        assert "max_jitter_score" in d
        assert d["min_confidence"] == 0.5


class TestQualityMetricCalculator:
    """QualityMetricCalculator 테스트"""
    
    @pytest.fixture
    def calculator(self):
        return QualityMetricCalculator()
    
    @pytest.fixture
    def good_poses(self):
        """좋은 품질의 포즈 데이터"""
        # [T, J, 4] - 100 frames, 33 joints, (x, y, z, visibility)
        np.random.seed(42)
        T, J = 100, 33
        
        # 부드러운 움직임 생성
        t = np.linspace(0, 2*np.pi, T)
        base_x = 0.5 + 0.1 * np.sin(t)
        base_y = 0.5 + 0.1 * np.cos(t)
        
        poses = np.zeros((T, J, 4))
        for j in range(J):
            poses[:, j, 0] = base_x + np.random.randn(T) * 0.01  # x
            poses[:, j, 1] = base_y + np.random.randn(T) * 0.01  # y
            poses[:, j, 2] = np.random.randn(T) * 0.01  # z
            poses[:, j, 3] = 0.9 + np.random.rand(T) * 0.1  # visibility (0.9~1.0)
        
        return poses.astype(np.float32)
    
    @pytest.fixture
    def bad_poses(self):
        """나쁜 품질의 포즈 데이터"""
        np.random.seed(42)
        T, J = 50, 33
        
        poses = np.zeros((T, J, 4))
        
        # 높은 지터 (급격한 변화)
        for j in range(J):
            poses[:, j, 0] = np.random.rand(T)  # 랜덤 x
            poses[:, j, 1] = np.random.rand(T)  # 랜덤 y
            poses[:, j, 2] = np.random.rand(T)  # 랜덤 z
            poses[:, j, 3] = 0.3 + np.random.rand(T) * 0.3  # 낮은 visibility
        
        # NaN 추가
        poses[10:15, :, :] = np.nan
        
        return poses.astype(np.float32)
    
    def test_calculate_good_quality(self, calculator, good_poses):
        """좋은 품질 데이터 분석"""
        metrics = calculator.calculate_pose_quality(good_poses)
        
        assert metrics.total_frames == 100
        assert metrics.confidence_mean > 0.8
        assert metrics.nan_ratio == 0.0
        assert metrics.jitter_score < 0.1
        assert metrics.quality_score > 0.5
    
    def test_calculate_bad_quality(self, calculator, bad_poses):
        """나쁜 품질 데이터 분석"""
        metrics = calculator.calculate_pose_quality(bad_poses)
        
        assert metrics.total_frames == 50
        assert metrics.nan_ratio > 0.05  # NaN 존재
        assert metrics.jitter_score > 0.1  # 높은 지터
        assert len(metrics.failure_reasons) > 0
    
    def test_empty_poses(self, calculator):
        """빈 포즈 데이터"""
        poses = np.zeros((0, 33, 4), dtype=np.float32)
        metrics = calculator.calculate_pose_quality(poses)
        
        assert metrics.total_frames == 0
    
    def test_nan_ratio_calculation(self, calculator):
        """NaN 비율 계산"""
        poses = np.ones((100, 33, 4), dtype=np.float32)
        poses[0:10, :, :] = np.nan  # 10% NaN
        
        metrics = calculator.calculate_pose_quality(poses)
        
        assert 0.09 <= metrics.nan_ratio <= 0.11
    
    def test_jitter_calculation(self, calculator):
        """Jitter 계산"""
        # 부드러운 움직임
        t = np.linspace(0, 2*np.pi, 100)
        smooth_poses = np.zeros((100, 33, 4))
        smooth_poses[:, 0, 0] = np.sin(t)
        smooth_poses[:, 0, 1] = np.cos(t)
        smooth_poses[:, :, 3] = 1.0  # visibility
        
        smooth_metrics = calculator.calculate_pose_quality(smooth_poses.astype(np.float32))
        
        # 급격한 움직임
        jerky_poses = np.zeros((100, 33, 4))
        jerky_poses[:, 0, 0] = np.random.rand(100)
        jerky_poses[:, 0, 1] = np.random.rand(100)
        jerky_poses[:, :, 3] = 1.0
        
        jerky_metrics = calculator.calculate_pose_quality(jerky_poses.astype(np.float32))
        
        # 부드러운 움직임이 더 낮은 지터
        assert smooth_metrics.jitter_score < jerky_metrics.jitter_score
    
    def test_quality_score_range(self, calculator, good_poses, bad_poses):
        """품질 점수 범위"""
        good_metrics = calculator.calculate_pose_quality(good_poses)
        bad_metrics = calculator.calculate_pose_quality(bad_poses)
        
        # 0.0 ~ 1.0 범위
        assert 0.0 <= good_metrics.quality_score <= 1.0
        assert 0.0 <= bad_metrics.quality_score <= 1.0
        
        # 좋은 데이터가 더 높은 점수
        assert good_metrics.quality_score > bad_metrics.quality_score
    
    def test_custom_thresholds(self, good_poses):
        """커스텀 임계값 적용"""
        # 엄격한 임계값
        strict = QualityThresholds(min_confidence=0.99, max_jitter_score=0.01)
        strict_calc = QualityMetricCalculator(strict)
        strict_metrics = strict_calc.calculate_pose_quality(good_poses)
        
        # 느슨한 임계값
        loose = QualityThresholds(min_confidence=0.1, max_jitter_score=1.0)
        loose_calc = QualityMetricCalculator(loose)
        loose_metrics = loose_calc.calculate_pose_quality(good_poses)
        
        # 느슨한 기준에서 더 많이 통과
        assert len(loose_metrics.failure_reasons) <= len(strict_metrics.failure_reasons)


class TestQualityFilterIntegration:
    """QualityFilter 통합 테스트"""
    
    @pytest.fixture
    def temp_poses_dir(self, tmp_path):
        """임시 포즈 디렉토리 생성"""
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        
        # 테스트 포즈 파일 생성
        for i in range(5):
            poses = np.random.rand(100, 33, 4).astype(np.float32)
            poses[:, :, 3] = 0.5 + 0.5 * np.random.rand(100, 33)  # visibility
            
            np.savez_compressed(
                poses_dir / f"video_{i}_pose.npz",
                poses=poses,
                confidences=np.mean(poses[:, :, 3], axis=1),
                fps=30.0,
            )
        
        return poses_dir
    
    def test_filter_imports(self):
        """Filter 모듈 임포트"""
        from filter_quality import QualityFilter, FilterResult
        assert QualityFilter is not None
        assert FilterResult is not None
    
    def test_analyze_file(self, temp_poses_dir):
        """단일 파일 분석"""
        from filter_quality import QualityFilter
        
        qf = QualityFilter(poses_dir=str(temp_poses_dir))
        
        file_path = temp_poses_dir / "video_0_pose.npz"
        result = qf.analyze_file(file_path)
        
        assert result.video_id == "video_0"
        assert result.total_frames == 100
        assert 0.0 <= result.quality_score <= 1.0
    
    def test_analyze_all(self, temp_poses_dir):
        """모든 파일 분석"""
        from filter_quality import QualityFilter
        
        qf = QualityFilter(poses_dir=str(temp_poses_dir))
        results = qf.analyze_all()
        
        assert len(results) == 5
    
    def test_filter_top_percent(self, temp_poses_dir, tmp_path):
        """상위 N% 필터링"""
        from filter_quality import QualityFilter
        
        filtered_dir = tmp_path / "filtered"
        qf = QualityFilter(
            poses_dir=str(temp_poses_dir),
            filtered_dir=str(filtered_dir),
        )
        
        results = qf.analyze_all()
        top_results = qf.filter_top_percent(results, top_percent=40, copy_files=True)
        
        # 5개 중 40% = 2개
        assert len(top_results) == 2
        
        # 파일이 복사되었는지 확인
        filtered_files = list(filtered_dir.glob("*.npz"))
        assert len(filtered_files) == 2
    
    def test_generate_report(self, temp_poses_dir, tmp_path):
        """리포트 생성"""
        from filter_quality import QualityFilter
        
        qf = QualityFilter(poses_dir=str(temp_poses_dir))
        results = qf.analyze_all()
        
        report_path = tmp_path / "report.json"
        report = qf.generate_report(results, str(report_path))
        
        assert "summary" in report
        assert "quality_stats" in report
        assert report["summary"]["total_files"] == 5
        
        # 파일 저장 확인
        assert report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
