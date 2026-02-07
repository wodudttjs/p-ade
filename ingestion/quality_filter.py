"""
비디오 품질 필터

기능:
- 기술적 품질 검사 (해상도, FPS, 길이)
- 컨텐츠 관련성 점수 계산
- 자동 거부/승인
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from ingestion.metadata_extractor import VideoMetadata
from core.logging_config import logger


class QualityLevel(Enum):
    """품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class QualityScore:
    """품질 점수"""
    technical_score: float  # 0.0 ~ 1.0
    relevance_score: float  # 0.0 ~ 1.0
    overall_score: float    # 가중 평균
    quality_level: QualityLevel
    
    # 세부 점수
    resolution_score: float = 0.0
    fps_score: float = 0.0
    duration_score: float = 0.0
    bitrate_score: float = 0.0
    
    # 거부 사유
    rejection_reasons: List[str] = field(default_factory=list)


class QualityFilter:
    """품질 필터"""
    
    # 기술 요구사항
    MIN_RESOLUTION_HEIGHT = 720  # 최소 720p
    PREFERRED_RESOLUTION_HEIGHT = 1080  # 선호 1080p
    
    MIN_FPS = 24
    PREFERRED_FPS = 30
    
    MIN_DURATION_SEC = 10
    MAX_DURATION_SEC = 1200  # 20분
    PREFERRED_DURATION_SEC = 300  # 5분
    
    MIN_BITRATE_MBPS = 1.0
    PREFERRED_BITRATE_MBPS = 5.0
    
    # 가중치
    TECHNICAL_WEIGHT = 0.6
    RELEVANCE_WEIGHT = 0.4
    
    # 승인 임계값
    REJECTION_THRESHOLD = 0.3
    ACCEPTABLE_THRESHOLD = 0.5
    GOOD_THRESHOLD = 0.7
    EXCELLENT_THRESHOLD = 0.85
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: 엄격 모드 (더 높은 기준)
        """
        self.strict_mode = strict_mode
        
        if strict_mode:
            self.MIN_RESOLUTION_HEIGHT = 1080
            self.MIN_FPS = 30
            self.ACCEPTABLE_THRESHOLD = 0.6
    
    def evaluate(
        self,
        metadata: VideoMetadata,
        video_info: Optional[Dict] = None,
    ) -> QualityScore:
        """
        비디오 품질 평가
        
        Args:
            metadata: 비디오 메타데이터
            video_info: 추가 정보 (제목, 설명, 태그 등)
        
        Returns:
            QualityScore
        """
        rejection_reasons = []
        
        # 1. 해상도 점수
        resolution_score = self._score_resolution(
            metadata.height,
            rejection_reasons
        )
        
        # 2. FPS 점수
        fps_score = self._score_fps(
            metadata.fps,
            rejection_reasons
        )
        
        # 3. 길이 점수
        duration_score = self._score_duration(
            metadata.duration_sec,
            rejection_reasons
        )
        
        # 4. 비트레이트 점수
        bitrate_mbps = metadata.bitrate_bps / 1_000_000
        bitrate_score = self._score_bitrate(
            bitrate_mbps,
            rejection_reasons
        )
        
        # 기술적 점수 (평균)
        technical_score = (
            resolution_score + fps_score + 
            duration_score + bitrate_score
        ) / 4.0
        
        # 5. 관련성 점수
        relevance_score = 1.0  # 기본값
        if video_info:
            relevance_score = self._score_relevance(video_info)
        
        # 종합 점수
        overall_score = (
            self.TECHNICAL_WEIGHT * technical_score +
            self.RELEVANCE_WEIGHT * relevance_score
        )
        
        # 등급 결정
        quality_level = self._determine_quality_level(
            overall_score,
            len(rejection_reasons) > 0
        )
        
        return QualityScore(
            technical_score=technical_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            quality_level=quality_level,
            resolution_score=resolution_score,
            fps_score=fps_score,
            duration_score=duration_score,
            bitrate_score=bitrate_score,
            rejection_reasons=rejection_reasons,
        )
    
    def _score_resolution(
        self,
        height: int,
        rejection_reasons: List[str]
    ) -> float:
        """해상도 점수"""
        if height < self.MIN_RESOLUTION_HEIGHT:
            rejection_reasons.append(
                f"Resolution too low: {height}p < {self.MIN_RESOLUTION_HEIGHT}p"
            )
            return 0.0
        
        if height >= self.PREFERRED_RESOLUTION_HEIGHT:
            return 1.0
        
        # 선형 보간
        score = (height - self.MIN_RESOLUTION_HEIGHT) / \
                (self.PREFERRED_RESOLUTION_HEIGHT - self.MIN_RESOLUTION_HEIGHT)
        
        return min(1.0, max(0.0, score))
    
    def _score_fps(self, fps: float, rejection_reasons: List[str]) -> float:
        """FPS 점수"""
        if fps < self.MIN_FPS:
            rejection_reasons.append(f"FPS too low: {fps} < {self.MIN_FPS}")
            return 0.0
        
        if fps >= self.PREFERRED_FPS:
            return 1.0
        
        score = (fps - self.MIN_FPS) / (self.PREFERRED_FPS - self.MIN_FPS)
        return min(1.0, max(0.0, score))
    
    def _score_duration(
        self,
        duration_sec: float,
        rejection_reasons: List[str]
    ) -> float:
        """길이 점수"""
        if duration_sec < self.MIN_DURATION_SEC:
            rejection_reasons.append(
                f"Video too short: {duration_sec}s < {self.MIN_DURATION_SEC}s"
            )
            return 0.0
        
        if duration_sec > self.MAX_DURATION_SEC:
            rejection_reasons.append(
                f"Video too long: {duration_sec}s > {self.MAX_DURATION_SEC}s"
            )
            return 0.0
        
        # 선호 길이에 가까울수록 높은 점수
        if duration_sec <= self.PREFERRED_DURATION_SEC:
            score = duration_sec / self.PREFERRED_DURATION_SEC
        else:
            # 너무 긴 경우 점수 감소
            excess = duration_sec - self.PREFERRED_DURATION_SEC
            max_excess = self.MAX_DURATION_SEC - self.PREFERRED_DURATION_SEC
            score = 1.0 - (excess / max_excess) * 0.5
        
        return min(1.0, max(0.0, score))
    
    def _score_bitrate(
        self,
        bitrate_mbps: float,
        rejection_reasons: List[str]
    ) -> float:
        """비트레이트 점수"""
        if bitrate_mbps < self.MIN_BITRATE_MBPS:
            rejection_reasons.append(
                f"Bitrate too low: {bitrate_mbps:.1f} Mbps"
            )
            return 0.2  # 완전 거부는 아님
        
        if bitrate_mbps >= self.PREFERRED_BITRATE_MBPS:
            return 1.0
        
        score = (bitrate_mbps - self.MIN_BITRATE_MBPS) / \
                (self.PREFERRED_BITRATE_MBPS - self.MIN_BITRATE_MBPS)
        
        return min(1.0, max(0.2, score))
    
    def _score_relevance(self, video_info: Dict) -> float:
        """컨텐츠 관련성 점수"""
        score = 0.0
        
        # 키워드 매칭
        keywords = self._get_target_keywords()
        
        title = video_info.get('title', '').lower()
        description = video_info.get('description', '').lower()
        tags = [t.lower() for t in video_info.get('tags', [])]
        
        # 제목 매칭 (가중치 높음)
        title_matches = sum(1 for kw in keywords if kw in title)
        score += (title_matches / len(keywords)) * 0.5
        
        # 태그 매칭
        tag_matches = sum(1 for kw in keywords if any(kw in tag for tag in tags))
        score += (tag_matches / len(keywords)) * 0.3
        
        # 설명 매칭
        desc_matches = sum(1 for kw in keywords if kw in description)
        score += (desc_matches / len(keywords)) * 0.2
        
        # 조회수/좋아요 고려 (인기도)
        views = video_info.get('view_count', 0)
        likes = video_info.get('like_count', 0)
        
        if views > 10000:
            score *= 1.1  # 보너스
        if likes > 100:
            score *= 1.05
        
        return min(1.0, score)
    
    @staticmethod
    def _get_target_keywords() -> List[str]:
        """타겟 키워드 (설정 파일에서 로드)"""
        return [
            'robot', 'assembly', 'manipulation',
            'grasping', 'pick', 'place',
            'tutorial', 'demonstration',
        ]
    
    def _determine_quality_level(
        self,
        overall_score: float,
        has_rejections: bool
    ) -> QualityLevel:
        """품질 등급 결정"""
        if has_rejections or overall_score < self.REJECTION_THRESHOLD:
            return QualityLevel.REJECTED
        elif overall_score >= self.EXCELLENT_THRESHOLD:
            return QualityLevel.EXCELLENT
        elif overall_score >= self.GOOD_THRESHOLD:
            return QualityLevel.GOOD
        elif overall_score >= self.ACCEPTABLE_THRESHOLD:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def should_process(self, quality_score: QualityScore) -> bool:
        """처리 여부 결정"""
        return quality_score.quality_level != QualityLevel.REJECTED


class QualityFilterPipeline:
    """필터 파이프라인"""
    
    def __init__(self, filter: QualityFilter):
        self.filter = filter
        self.stats = {
            'total': 0,
            'accepted': 0,
            'rejected': 0,
            'by_level': {level: 0 for level in QualityLevel},
        }
    
    def process(
        self,
        metadata: VideoMetadata,
        video_info: Optional[Dict] = None,
    ) -> QualityScore:
        """비디오 처리"""
        self.stats['total'] += 1
        
        score = self.filter.evaluate(metadata, video_info)
        
        self.stats['by_level'][score.quality_level] += 1
        
        if self.filter.should_process(score):
            self.stats['accepted'] += 1
        else:
            self.stats['rejected'] += 1
        
        return score
    
    def get_statistics(self) -> Dict:
        """통계 조회"""
        acceptance_rate = (
            self.stats['accepted'] / self.stats['total']
            if self.stats['total'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'acceptance_rate': acceptance_rate,
        }
    
    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()
        
        logger.info("\n=== Quality Filter Statistics ===")
        logger.info(f"Total Videos: {stats['total']}")
        logger.info(f"Accepted: {stats['accepted']} ({stats['acceptance_rate']:.1%})")
        logger.info(f"Rejected: {stats['rejected']}")
        logger.info("\nBy Quality Level:")
        for level, count in stats['by_level'].items():
            if count > 0:
                logger.info(f"  {level.value}: {count}")
    
    def generate_report(self, output_file: str):
        """주간 리포트 생성"""
        import json
        from datetime import datetime
        
        stats = self.get_statistics()
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'statistics': stats,
            'quality_distribution': {
                level.value: count 
                for level, count in stats['by_level'].items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_file}")
