"""
Quality Metric 계산 모듈

Feedback #5: Quality filtering 기준 동적 관리
- 임계값을 코드 상수로 박지 않고 DB/Config로 관리
- 주간 리포트에서 "기준 변경 시 예상 통과율" 제공
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import numpy as np
import json

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class QualityThresholds:
    """
    품질 임계값 설정
    
    DB의 QualityConfig 테이블과 연동
    """
    # 신뢰도
    min_confidence: float = 0.5
    
    # Jitter
    max_jitter_score: float = 0.3
    
    # 에피소드 길이
    min_episode_frames: int = 30
    
    # NaN 비율
    max_nan_ratio: float = 0.1
    
    # 포즈 품질
    min_visible_joints: int = 15
    min_pose_completeness: float = 0.7
    
    # 프로파일
    profile: str = "default"  # dev, prod, strict
    
    @classmethod
    def from_db(cls, db_session, profile: str = "default") -> "QualityThresholds":
        """DB에서 설정 로드"""
        try:
            from models.database import QualityConfig
            
            config = db_session.query(QualityConfig).filter(
                QualityConfig.profile == profile,
                QualityConfig.is_active == True
            ).first()
            
            if config:
                return cls(
                    min_confidence=config.min_confidence,
                    max_jitter_score=config.max_jitter_score,
                    min_episode_frames=config.min_episode_frames,
                    max_nan_ratio=config.max_nan_ratio,
                    min_visible_joints=config.min_visible_joints,
                    min_pose_completeness=config.min_pose_completeness,
                    profile=profile,
                )
        except Exception as e:
            logger.warning(f"DB 설정 로드 실패, 기본값 사용: {e}")
        
        return cls(profile=profile)
    
    @classmethod
    def from_config(cls, config_path: str) -> "QualityThresholds":
        """Config 파일에서 로드"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PoseQualityMetrics:
    """포즈 품질 메트릭"""
    # 기본 메트릭
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    confidence_min: float = 0.0
    
    # NaN 관련
    nan_ratio: float = 0.0
    nan_frames: int = 0
    total_frames: int = 0
    
    # Jitter
    jitter_score: float = 0.0
    jitter_per_joint: List[float] = field(default_factory=list)
    
    # 가시성
    visible_joints_mean: float = 0.0
    pose_completeness: float = 0.0
    
    # 연속성
    continuous_frames: int = 0
    max_gap_frames: int = 0
    
    # 품질 등급
    quality_score: float = 0.0  # 0.0 ~ 1.0
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


class QualityMetricCalculator:
    """
    품질 메트릭 계산기
    
    Feedback #4: quality metric 계산을 모듈화 (입력/출력 고정)
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
    
    def calculate_pose_quality(
        self,
        poses: np.ndarray,  # [T, J, 4] with visibility
        confidence: Optional[np.ndarray] = None,  # [T]
    ) -> PoseQualityMetrics:
        """
        포즈 품질 메트릭 계산
        
        Args:
            poses: [T, J, 4] 배열 (x, y, z, visibility)
            confidence: [T] 프레임별 신뢰도
        
        Returns:
            PoseQualityMetrics
        """
        metrics = PoseQualityMetrics()
        failure_reasons = []
        
        T = poses.shape[0]
        metrics.total_frames = T
        
        # 1. NaN 분석
        nan_mask = np.isnan(poses).any(axis=(1, 2))
        metrics.nan_frames = int(np.sum(nan_mask))
        metrics.nan_ratio = metrics.nan_frames / T if T > 0 else 0.0
        
        if metrics.nan_ratio > self.thresholds.max_nan_ratio:
            failure_reasons.append(f"nan_ratio={metrics.nan_ratio:.2f} > {self.thresholds.max_nan_ratio}")
        
        # 2. 신뢰도 분석
        if confidence is not None:
            valid_conf = confidence[~np.isnan(confidence)]
            if len(valid_conf) > 0:
                metrics.confidence_mean = float(np.mean(valid_conf))
                metrics.confidence_std = float(np.std(valid_conf))
                metrics.confidence_min = float(np.min(valid_conf))
        else:
            # visibility에서 추출
            if poses.shape[2] >= 4:
                visibility = poses[:, :, 3]
                valid_vis = visibility[~np.isnan(visibility)]
                if len(valid_vis) > 0:
                    metrics.confidence_mean = float(np.mean(valid_vis))
                    metrics.confidence_std = float(np.std(valid_vis))
        
        if metrics.confidence_mean < self.thresholds.min_confidence:
            failure_reasons.append(f"confidence={metrics.confidence_mean:.2f} < {self.thresholds.min_confidence}")
        
        # 3. Jitter 계산
        metrics.jitter_score = self._calculate_jitter(poses[:, :, :3])
        
        if metrics.jitter_score > self.thresholds.max_jitter_score:
            failure_reasons.append(f"jitter={metrics.jitter_score:.2f} > {self.thresholds.max_jitter_score}")
        
        # 4. 가시성 분석
        if poses.shape[2] >= 4:
            visibility = poses[:, :, 3]
            visible_per_frame = np.sum(visibility > 0.5, axis=1)
            metrics.visible_joints_mean = float(np.mean(visible_per_frame))
            metrics.pose_completeness = metrics.visible_joints_mean / poses.shape[1]
            
            if metrics.visible_joints_mean < self.thresholds.min_visible_joints:
                failure_reasons.append(f"visible_joints={metrics.visible_joints_mean:.1f} < {self.thresholds.min_visible_joints}")
        
        # 5. 연속성 분석
        metrics.continuous_frames, metrics.max_gap_frames = self._analyze_continuity(nan_mask)
        
        if metrics.continuous_frames < self.thresholds.min_episode_frames:
            failure_reasons.append(f"continuous_frames={metrics.continuous_frames} < {self.thresholds.min_episode_frames}")
        
        # 6. 종합 품질 점수 계산
        metrics.quality_score = self._calculate_quality_score(metrics)
        metrics.failure_reasons = failure_reasons
        metrics.passed = len(failure_reasons) == 0
        
        return metrics
    
    def _calculate_jitter(self, poses: np.ndarray) -> float:
        """
        Jitter 계산 (프레임 간 위치 변화의 분산)
        
        낮을수록 좋음 (부드러운 움직임)
        """
        if poses.shape[0] < 3:
            return 0.0
        
        # 프레임 간 속도 (1차 미분)
        velocity = np.diff(poses, axis=0)
        
        # 속도의 변화 (2차 미분 = 가속도)
        acceleration = np.diff(velocity, axis=0)
        
        # Jitter = 가속도의 평균 크기
        jitter = np.nanmean(np.abs(acceleration))
        
        return float(jitter) if not np.isnan(jitter) else 0.0
    
    def _analyze_continuity(self, nan_mask: np.ndarray) -> tuple:
        """
        연속성 분석
        
        Returns:
            (최대 연속 프레임 수, 최대 갭)
        """
        if len(nan_mask) == 0:
            return 0, 0
        
        # 연속된 valid 프레임 찾기
        valid = ~nan_mask
        
        # Run-length encoding
        max_continuous = 0
        current_run = 0
        
        for v in valid:
            if v:
                current_run += 1
                max_continuous = max(max_continuous, current_run)
            else:
                current_run = 0
        
        # 최대 갭
        max_gap = 0
        current_gap = 0
        
        for v in valid:
            if not v:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        
        return max_continuous, max_gap
    
    def _calculate_quality_score(self, metrics: PoseQualityMetrics) -> float:
        """종합 품질 점수 계산 (0.0 ~ 1.0)"""
        score = 0.0
        weights = {
            "confidence": 0.3,
            "nan": 0.2,
            "jitter": 0.2,
            "completeness": 0.15,
            "continuity": 0.15,
        }
        
        # 신뢰도 점수
        conf_score = min(metrics.confidence_mean / self.thresholds.min_confidence, 1.0)
        score += weights["confidence"] * conf_score
        
        # NaN 점수 (낮을수록 좋음)
        nan_score = 1.0 - min(metrics.nan_ratio / self.thresholds.max_nan_ratio, 1.0)
        score += weights["nan"] * nan_score
        
        # Jitter 점수 (낮을수록 좋음)
        jitter_score = 1.0 - min(metrics.jitter_score / self.thresholds.max_jitter_score, 1.0)
        score += weights["jitter"] * jitter_score
        
        # 완성도 점수
        comp_score = min(metrics.pose_completeness / self.thresholds.min_pose_completeness, 1.0)
        score += weights["completeness"] * comp_score
        
        # 연속성 점수
        if metrics.total_frames > 0:
            cont_score = metrics.continuous_frames / metrics.total_frames
        else:
            cont_score = 0.0
        score += weights["continuity"] * cont_score
        
        return min(max(score, 0.0), 1.0)
    
    def estimate_pass_rate(
        self,
        sample_data: List[np.ndarray],
        new_thresholds: Optional[QualityThresholds] = None
    ) -> Dict[str, float]:
        """
        새로운 임계값에서의 예상 통과율 계산
        
        Feedback #5: 주간 리포트에서 "기준 변경 시 예상 통과율" 제공
        """
        if new_thresholds:
            calc = QualityMetricCalculator(new_thresholds)
        else:
            calc = self
        
        total = len(sample_data)
        passed = 0
        failure_stats: Dict[str, int] = {}
        
        for poses in sample_data:
            metrics = calc.calculate_pose_quality(poses)
            if metrics.passed:
                passed += 1
            else:
                for reason in metrics.failure_reasons:
                    key = reason.split("=")[0]
                    failure_stats[key] = failure_stats.get(key, 0) + 1
        
        pass_rate = passed / total if total > 0 else 0.0
        
        return {
            "total_samples": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "failure_stats": failure_stats,
            "thresholds": asdict(new_thresholds or self.thresholds),
        }


class QualityDistributionTracker:
    """
    품질 분포 추적 (모니터링용)
    
    Feedback #10: 데이터 품질 분포 대시보드
    - conf 평균/분산
    - jitter 분포
    - 에피소드 길이 분포
    - 카테고리별 균형
    """
    
    def __init__(self):
        self._metrics: List[PoseQualityMetrics] = []
        self._categories: Dict[str, int] = {}
    
    def add_sample(self, metrics: PoseQualityMetrics, category: Optional[str] = None):
        """샘플 추가"""
        self._metrics.append(metrics)
        if category:
            self._categories[category] = self._categories.get(category, 0) + 1
    
    def get_distribution_summary(self) -> Dict[str, Any]:
        """분포 요약"""
        if not self._metrics:
            return {}
        
        confidences = [m.confidence_mean for m in self._metrics]
        jitters = [m.jitter_score for m in self._metrics]
        nan_ratios = [m.nan_ratio for m in self._metrics]
        quality_scores = [m.quality_score for m in self._metrics]
        frame_counts = [m.total_frames for m in self._metrics]
        
        return {
            "total_samples": len(self._metrics),
            "passed_count": sum(1 for m in self._metrics if m.passed),
            "pass_rate": sum(1 for m in self._metrics if m.passed) / len(self._metrics),
            
            "confidence": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
            },
            
            "jitter": {
                "mean": float(np.mean(jitters)),
                "std": float(np.std(jitters)),
                "percentile_50": float(np.percentile(jitters, 50)),
                "percentile_95": float(np.percentile(jitters, 95)),
            },
            
            "nan_ratio": {
                "mean": float(np.mean(nan_ratios)),
                "max": float(np.max(nan_ratios)),
            },
            
            "quality_score": {
                "mean": float(np.mean(quality_scores)),
                "std": float(np.std(quality_scores)),
                "percentile_25": float(np.percentile(quality_scores, 25)),
                "percentile_50": float(np.percentile(quality_scores, 50)),
                "percentile_75": float(np.percentile(quality_scores, 75)),
            },
            
            "episode_length": {
                "mean": float(np.mean(frame_counts)),
                "std": float(np.std(frame_counts)),
                "min": int(np.min(frame_counts)),
                "max": int(np.max(frame_counts)),
            },
            
            "category_balance": self._categories,
        }
    
    def to_prometheus_metrics(self) -> Dict[str, float]:
        """Prometheus 메트릭 형식으로 변환"""
        summary = self.get_distribution_summary()
        if not summary:
            return {}
        
        return {
            "quality_pass_rate": summary["pass_rate"],
            "quality_confidence_mean": summary["confidence"]["mean"],
            "quality_confidence_std": summary["confidence"]["std"],
            "quality_jitter_mean": summary["jitter"]["mean"],
            "quality_jitter_p95": summary["jitter"]["percentile_95"],
            "quality_nan_ratio_mean": summary["nan_ratio"]["mean"],
            "quality_score_mean": summary["quality_score"]["mean"],
            "quality_episode_length_mean": summary["episode_length"]["mean"],
        }
