#!/usr/bin/env python
"""
Quality Filtering CLI

MVP Phase 2 Week 6: Quality Filtering
- Confidence score 필터링
- Jittering 점수 계산
- 상위 50% 데이터만 저장

사용법:
    python filter_quality.py --all                     # 모든 포즈 파일 필터링
    python filter_quality.py --file data/poses/x.npz  # 단일 파일 분석
    python filter_quality.py --analyze                 # 품질 분석 리포트
    python filter_quality.py --top-percent 50          # 상위 50% 선택
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import shutil

import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from core.quality_metrics import (
    QualityMetricCalculator,
    QualityThresholds,
    PoseQualityMetrics,
)

logger = setup_logger(__name__)


@dataclass
class FilterResult:
    """필터링 결과"""
    file_path: str
    video_id: str
    passed: bool
    quality_score: float
    confidence_mean: float
    jitter_score: float
    nan_ratio: float
    total_frames: int
    failure_reasons: List[str]


class QualityFilter:
    """품질 필터링 클래스"""
    
    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
        poses_dir: str = "data/poses",
        filtered_dir: str = "data/filtered",
    ):
        self.thresholds = thresholds or QualityThresholds()
        self.calculator = QualityMetricCalculator(self.thresholds)
        self.poses_dir = Path(poses_dir)
        self.filtered_dir = Path(filtered_dir)
    
    def analyze_file(self, file_path: Path) -> FilterResult:
        """단일 파일 분석"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # 포즈 데이터 추출 (여러 키 지원)
            poses = None
            for key in ["poses", "body", "keypoints"]:
                if key in data:
                    poses = data[key]
                    break
            
            if poses is None:
                raise ValueError(f"포즈 데이터 없음 (keys: {list(data.keys())})")
            
            # [T, J*4] -> [T, J, 4] 변환 (필요시)
            if len(poses.shape) == 2:
                num_joints = poses.shape[1] // 4
                poses = poses.reshape(-1, num_joints, 4)
            
            # [T, J, 3] -> [T, J, 4] 변환 (visibility 추가)
            if len(poses.shape) == 3 and poses.shape[2] == 3:
                # visibility가 별도 배열로 있는지 확인
                visibility = np.ones((poses.shape[0], poses.shape[1]), dtype=np.float32)
                poses_with_vis = np.concatenate([
                    poses,
                    visibility[:, :, np.newaxis]
                ], axis=2)
                poses = poses_with_vis
            
            # 신뢰도 데이터 (여러 키 지원)
            confidences = None
            for key in ["confidences", "confidence", "conf"]:
                if key in data:
                    confidences = data[key]
                    break
            
            # 품질 메트릭 계산
            metrics = self.calculator.calculate_pose_quality(poses, confidences)
            
            # 비디오 ID 추출
            video_id = file_path.stem.replace("_pose", "")
            
            return FilterResult(
                file_path=str(file_path),
                video_id=video_id,
                passed=metrics.passed,
                quality_score=metrics.quality_score,
                confidence_mean=metrics.confidence_mean,
                jitter_score=metrics.jitter_score,
                nan_ratio=metrics.nan_ratio,
                total_frames=metrics.total_frames,
                failure_reasons=metrics.failure_reasons,
            )
            
        except Exception as e:
            logger.error(f"파일 분석 실패 {file_path}: {e}")
            return FilterResult(
                file_path=str(file_path),
                video_id=file_path.stem.replace("_pose", ""),
                passed=False,
                quality_score=0.0,
                confidence_mean=0.0,
                jitter_score=0.0,
                nan_ratio=1.0,
                total_frames=0,
                failure_reasons=[f"분석 오류: {str(e)}"],
            )
    
    def analyze_all(self) -> List[FilterResult]:
        """모든 포즈 파일 분석"""
        results = []
        
        pose_files = list(self.poses_dir.glob("*_pose.npz"))
        
        if not pose_files:
            logger.warning(f"포즈 파일 없음: {self.poses_dir}")
            return results
        
        print(f"\n{'='*60}")
        print(f"🔍 품질 분석 시작")
        print(f"{'='*60}")
        print(f"📁 경로: {self.poses_dir}")
        print(f"📦 파일: {len(pose_files)}개")
        print()
        
        for i, file_path in enumerate(pose_files, 1):
            result = self.analyze_file(file_path)
            results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"[{i}/{len(pose_files)}] {result.video_id}: {status} "
                  f"(score={result.quality_score:.2f}, conf={result.confidence_mean:.2f})")
        
        return results
    
    def filter_by_threshold(
        self,
        results: Optional[List[FilterResult]] = None,
        copy_files: bool = True,
    ) -> Tuple[List[FilterResult], List[FilterResult]]:
        """임계값 기반 필터링"""
        if results is None:
            results = self.analyze_all()
        
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        if copy_files and passed:
            self.filtered_dir.mkdir(parents=True, exist_ok=True)
            
            for result in passed:
                src = Path(result.file_path)
                dst = self.filtered_dir / src.name
                shutil.copy2(src, dst)
        
        return passed, failed
    
    def filter_top_percent(
        self,
        results: Optional[List[FilterResult]] = None,
        top_percent: float = 50.0,
        copy_files: bool = True,
    ) -> List[FilterResult]:
        """상위 N% 선택"""
        if results is None:
            results = self.analyze_all()
        
        # 품질 점수로 정렬
        sorted_results = sorted(results, key=lambda r: r.quality_score, reverse=True)
        
        # 상위 N% 선택
        top_count = max(1, int(len(sorted_results) * (top_percent / 100)))
        top_results = sorted_results[:top_count]
        
        if copy_files and top_results:
            self.filtered_dir.mkdir(parents=True, exist_ok=True)
            
            for result in top_results:
                src = Path(result.file_path)
                dst = self.filtered_dir / src.name
                shutil.copy2(src, dst)
        
        return top_results
    
    def generate_report(
        self,
        results: List[FilterResult],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """품질 리포트 생성"""
        if not results:
            return {"error": "분석 결과 없음"}
        
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        # 통계 계산
        quality_scores = [r.quality_score for r in results]
        confidence_means = [r.confidence_mean for r in results]
        jitter_scores = [r.jitter_score for r in results]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "thresholds": asdict(self.thresholds),
            "summary": {
                "total_files": len(results),
                "passed": len(passed),
                "failed": len(failed),
                "pass_rate": len(passed) / len(results) * 100 if results else 0,
            },
            "quality_stats": {
                "score_mean": float(np.mean(quality_scores)),
                "score_std": float(np.std(quality_scores)),
                "score_min": float(np.min(quality_scores)),
                "score_max": float(np.max(quality_scores)),
                "score_median": float(np.median(quality_scores)),
            },
            "confidence_stats": {
                "mean": float(np.mean(confidence_means)),
                "std": float(np.std(confidence_means)),
                "min": float(np.min(confidence_means)),
                "max": float(np.max(confidence_means)),
            },
            "jitter_stats": {
                "mean": float(np.mean(jitter_scores)),
                "std": float(np.std(jitter_scores)),
                "min": float(np.min(jitter_scores)),
                "max": float(np.max(jitter_scores)),
            },
            "failure_analysis": self._analyze_failures(failed),
            "passed_files": [asdict(r) for r in passed],
            "failed_files": [asdict(r) for r in failed],
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"리포트 저장: {output_path}")
        
        return report
    
    def _analyze_failures(self, failed: List[FilterResult]) -> Dict[str, int]:
        """실패 원인 분석"""
        reasons = {}
        for result in failed:
            for reason in result.failure_reasons:
                # 원인 카테고리 추출
                category = reason.split("=")[0] if "=" in reason else reason
                reasons[category] = reasons.get(category, 0) + 1
        return reasons
    
    def print_summary(self, results: List[FilterResult]):
        """요약 출력"""
        if not results:
            print("분석 결과 없음")
            return
        
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        print()
        print("="*60)
        print("📊 품질 필터링 결과")
        print("="*60)
        
        print(f"\n📈 요약:")
        print(f"   총 파일: {len(results)}개")
        print(f"   ✅ 통과: {len(passed)}개 ({len(passed)/len(results)*100:.1f}%)")
        print(f"   ❌ 실패: {len(failed)}개 ({len(failed)/len(results)*100:.1f}%)")
        
        if results:
            scores = [r.quality_score for r in results]
            print(f"\n📊 품질 점수:")
            print(f"   평균: {np.mean(scores):.3f}")
            print(f"   표준편차: {np.std(scores):.3f}")
            print(f"   최소: {np.min(scores):.3f}")
            print(f"   최대: {np.max(scores):.3f}")
        
        if failed:
            print(f"\n❌ 실패 원인:")
            failure_analysis = self._analyze_failures(failed)
            for reason, count in sorted(failure_analysis.items(), key=lambda x: -x[1]):
                print(f"   - {reason}: {count}개")
        
        print()


def update_database_quality(
    results: List[FilterResult],
    db_path: str = "data/pade.db"
):
    """DB에 품질 점수 업데이트"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.database import Episode
        
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        updated = 0
        for result in results:
            # video_id로 에피소드 찾기
            episodes = session.query(Episode).filter(
                Episode.episode_id.like(f"{result.video_id}%")
            ).all()
            
            for ep in episodes:
                ep.quality_score = result.quality_score
                ep.confidence_score = result.confidence_mean
                ep.jittering_score = result.jitter_score
                updated += 1
        
        session.commit()
        session.close()
        
        logger.info(f"DB 업데이트: {updated}개 에피소드")
        return updated
        
    except Exception as e:
        logger.error(f"DB 업데이트 실패: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="P-ADE 품질 필터링")
    
    parser.add_argument("--all", action="store_true", help="모든 포즈 파일 분석")
    parser.add_argument("--file", help="단일 파일 분석")
    parser.add_argument("--analyze", action="store_true", help="분석만 수행 (필터링 없음)")
    parser.add_argument("--top-percent", type=float, default=None, help="상위 N%% 선택")
    
    parser.add_argument("--poses-dir", default="data/poses", help="포즈 디렉토리")
    parser.add_argument("--output-dir", default="data/filtered", help="출력 디렉토리")
    parser.add_argument("--report", default=None, help="리포트 저장 경로")
    
    # 임계값 옵션
    parser.add_argument("--min-confidence", type=float, default=0.5, help="최소 신뢰도")
    parser.add_argument("--max-jitter", type=float, default=0.3, help="최대 지터")
    parser.add_argument("--min-frames", type=int, default=30, help="최소 프레임 수")
    
    parser.add_argument("--update-db", action="store_true", help="DB 업데이트")
    parser.add_argument("--dry-run", action="store_true", help="파일 복사 없이 분석만")
    parser.add_argument("--no-gpu-streams", action="store_true", help="GPU 3-Stream 비활성화")
    
    args = parser.parse_args()
    
    # 임계값 설정
    thresholds = QualityThresholds(
        min_confidence=args.min_confidence,
        max_jitter_score=args.max_jitter,
        min_episode_frames=args.min_frames,
    )
    
    # 필터 생성
    qf = QualityFilter(
        thresholds=thresholds,
        poses_dir=args.poses_dir,
        filtered_dir=args.output_dir,
    )
    
    results = []
    
    if args.file:
        # 단일 파일 분석
        result = qf.analyze_file(Path(args.file))
        results = [result]
        
        print(f"\n📄 파일: {result.file_path}")
        print(f"   Video ID: {result.video_id}")
        print(f"   통과: {'✅ YES' if result.passed else '❌ NO'}")
        print(f"   품질 점수: {result.quality_score:.3f}")
        print(f"   신뢰도: {result.confidence_mean:.3f}")
        print(f"   지터: {result.jitter_score:.3f}")
        print(f"   NaN 비율: {result.nan_ratio:.3f}")
        print(f"   프레임: {result.total_frames}")
        
        if result.failure_reasons:
            print(f"   실패 원인:")
            for reason in result.failure_reasons:
                print(f"     - {reason}")
    
    elif args.all or args.analyze:
        # GPU 3-Stream 병렬 처리 시도
        gpu_used = False
        if not args.no_gpu_streams and not args.analyze:
            try:
                from gpu.stream_manager import GPU3StreamManager
                stream_mgr = GPU3StreamManager()
                batch_size = stream_mgr.auto_adjust_batch_size()
                vram = stream_mgr.get_vram_usage()
                print(f"\n🎮 GPU 3-Stream 활성화 (배치: {batch_size}, VRAM: {vram.get('allocated', 0):.1f}GB)")
                
                pose_files = list(Path(args.poses_dir).glob("*_pose.npz"))
                if pose_files:
                    processor = stream_mgr.make_quality_filter_processor(
                        filtered_dir=None if args.dry_run else args.output_dir,
                        threshold=thresholds.min_confidence,
                    )
                    gpu_results = stream_mgr.process_batch(
                        [str(f) for f in pose_files], processor
                    )
                    
                    for r in gpu_results:
                        if r and r.get("success"):
                            passed_str = "✅ PASS" if r.get("passed") else "❌ FAIL"
                            print(f"  {passed_str} {r.get('video_id', '?')}: "
                                  f"score={r.get('quality_score', 0):.2f}, conf={r.get('confidence_mean', 0):.2f}")
                            results.append(FilterResult(
                                file_path=r.get("file_path", ""),
                                video_id=r.get("video_id", ""),
                                passed=r.get("passed", False),
                                quality_score=r.get("quality_score", 0.0),
                                confidence_mean=r.get("confidence_mean", 0.0),
                                jitter_score=r.get("jitter_score", 0.0),
                                nan_ratio=0.0,
                                total_frames=0,
                                failure_reasons=[],
                            ))
                        else:
                            print(f"  ❌ {r.get('video_id', '?')}: {r.get('error', 'unknown')}")
                    
                    stream_mgr.print_stats()
                    gpu_used = True
            except Exception as e:
                print(f"⚠️ GPU 3-Stream 실패, 순차 모드로 폴백: {e}")
        
        # 폴백: 순차 처리
        if not gpu_used:
            results = qf.analyze_all()
        
        if args.top_percent:
            # 상위 N% 선택
            top_results = qf.filter_top_percent(
                results,
                top_percent=args.top_percent,
                copy_files=not args.dry_run,
            )
            print(f"\n🏆 상위 {args.top_percent}% 선택: {len(top_results)}개")
            
            if not args.dry_run:
                print(f"   저장 경로: {qf.filtered_dir}")
        
        elif not args.analyze:
            # 임계값 기반 필터링
            passed, failed = qf.filter_by_threshold(
                results,
                copy_files=not args.dry_run,
            )
            
            if not args.dry_run and passed:
                print(f"\n📁 필터링된 파일 저장: {qf.filtered_dir}")
        
        # 요약 출력
        qf.print_summary(results)
        
        # 리포트 저장
        if args.report:
            report = qf.generate_report(results, args.report)
            print(f"📝 리포트 저장: {args.report}")
    
    else:
        parser.print_help()
        return
    
    # DB 업데이트
    if args.update_db and results:
        updated = update_database_quality(results)
        print(f"💾 DB 업데이트: {updated}개 에피소드")


if __name__ == "__main__":
    main()
