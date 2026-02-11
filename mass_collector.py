#!/usr/bin/env python
"""
P-ADE 대량 수집 오케스트레이터

500개 이상의 로봇팔 영상을 자동으로 수집하는 end-to-end 파이프라인입니다.

파이프라인 단계:
  1. 키워드 생성 → 다중 소스 크롤링 (URL 수집)
  2. 병렬 다운로드 (비디오 파일 저장)
  3. 객체 검출 & Episode 생성
  4. S3 클라우드 업로드
  5. 통계 리포트 출력

사용법:
    # 전체 파이프라인 실행 (500개 목표)
    python mass_collector.py --target 500

    # 크롤링만 실행
    python mass_collector.py --target 500 --stage crawl

    # 다운로드부터 재시작
    python mass_collector.py --target 500 --stage download

    # 특정 단계만 실행
    python mass_collector.py --stage detect --limit 100
    python mass_collector.py --stage upload

    # 커스텀 키워드로 실행
    python mass_collector.py --target 200 --keywords "robot arm,pick and place,cobot"

    # 드라이런 (실제 실행 없이 계획만 출력)
    python mass_collector.py --target 500 --dry-run
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

# 프로젝트 루트 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from config.settings import Config

logger = setup_logger(__name__)


# ============================================================
# 파이프라인 설정
# ============================================================

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 수집 목표
    target_count: int = 500
    
    # 크롤링 설정
    sources: List[str] = field(default_factory=lambda: ["youtube", "google_videos"])
    languages: List[str] = field(default_factory=lambda: ["en", "ko"])
    crawl_workers: int = 4
    crawl_full_info: bool = False
    min_duration_sec: int = 30
    max_duration_sec: int = 1200
    content_filter: bool = True
    use_multiprocess: bool = False  # 멀티프로세스 크롤링 모드
    use_async: bool = False  # 비동기 크롤링 모드
    
    # 다운로드 설정
    download_workers: int = 6
    download_timeout: int = 600
    download_quality: str = "720p"
    
    # 검출 설정
    detect_fps: float = 5.0
    detect_device: Optional[str] = None  # None = auto-detect
    detect_batch_size: int = 50
    use_gpu_streams: bool = True  # GPU 3-stream 병렬 처리
    
    # 품질 평가 설정
    quality_filter: bool = True  # 품질 필터링 활성화
    quality_threshold: float = 60.0  # 통과 점수 (100점 만점)
    
    # 업로드 설정
    s3_bucket: str = ""
    s3_prefix: str = "episodes"
    upload_workers: int = 4
    
    # 경로
    db_path: str = "data/pade.db"
    raw_dir: str = "data/raw"
    episodes_dir: str = "data/episodes"
    urls_csv: str = "data/urls_mass.csv"
    report_path: str = "data/collection_report.json"
    
    # 기타
    dry_run: bool = False
    resume: bool = True  # 이전 진행 이어받기
    
    @property
    def crawl_multiplier(self) -> float:
        """목표 대비 크롤링 초과 수집 배수 (필터링 감안)"""
        return 3.0
    
    @property
    def crawl_target(self) -> int:
        """실제 크롤링 목표 (필터/중복 감안)"""
        return int(self.target_count * self.crawl_multiplier)


@dataclass
class StageResult:
    """단계별 결과"""
    stage: str
    success: bool
    count: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"{status} [{self.stage}] "
            f"완료: {self.count}개, 오류: {self.errors}개, "
            f"소요: {self.elapsed_sec:.1f}초"
        )


@dataclass
class PipelineReport:
    """파이프라인 전체 리포트"""
    started_at: str = ""
    completed_at: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    total_crawled: int = 0
    total_downloaded: int = 0
    total_episodes: int = 0
    total_uploaded: int = 0
    total_elapsed_sec: float = 0.0
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        print()
        print("=" * 70)
        print("📋 P-ADE 대량 수집 파이프라인 리포트")
        print("=" * 70)
        print(f"  시작: {self.started_at}")
        print(f"  완료: {self.completed_at}")
        print(f"  총 소요: {self.total_elapsed_sec:.1f}초 ({self.total_elapsed_sec/60:.1f}분)")
        print()
        print(f"  📊 결과:")
        print(f"    크롤링 URL: {self.total_crawled}개")
        print(f"    다운로드:   {self.total_downloaded}개")
        print(f"    에피소드:   {self.total_episodes}개")
        print(f"    업로드:     {self.total_uploaded}개")
        print()
        for stage in self.stages:
            sr = StageResult(**stage)
            print(f"  {sr.summary()}")
        print("=" * 70)


# ============================================================
# 파이프라인 실행기
# ============================================================

class MassCollector:
    """대량 수집 파이프라인 오케스트레이터"""

    STAGES = ["crawl", "download", "detect", "quality", "upload"]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.report = PipelineReport(
            started_at=datetime.now().isoformat(),
            config=asdict(config),
        )

    def run(self, start_stage: Optional[str] = None, end_stage: Optional[str] = None):
        """파이프라인 실행"""
        stages = self.STAGES.copy()

        # 시작/종료 단계 설정
        if start_stage:
            start_idx = stages.index(start_stage) if start_stage in stages else 0
            stages = stages[start_idx:]
        if end_stage:
            end_idx = stages.index(end_stage) + 1 if end_stage in stages else len(stages)
            stages = stages[:end_idx]

        total_start = time.time()

        print()
        print("🚀" + "=" * 68)
        print("   P-ADE 대량 수집 파이프라인")
        print("=" * 70)
        print(f"   목표: {self.config.target_count}개 영상")
        print(f"   소스: {', '.join(self.config.sources)}")
        print(f"   단계: {' → '.join(stages)}")
        print(f"   드라이런: {'예' if self.config.dry_run else '아니오'}")
        print("=" * 70)

        for stage_name in stages:
            print(f"\n{'─'*70}")
            print(f"📌 단계: {stage_name.upper()}")
            print(f"{'─'*70}")

            handler = getattr(self, f"_stage_{stage_name}", None)
            if not handler:
                logger.error(f"알 수 없는 단계: {stage_name}")
                continue

            try:
                result = handler()
                self.report.stages.append(asdict(result))
                print(f"  {result.summary()}")
            except Exception as e:
                logger.error(f"단계 실패 [{stage_name}]: {e}")
                result = StageResult(
                    stage=stage_name, success=False, errors=1,
                    details={"error": str(e)},
                )
                self.report.stages.append(asdict(result))
                print(f"  {result.summary()}")
                # 크롤링/다운로드 실패 시에도 계속 진행
                if stage_name in ("crawl",):
                    print("  ⚠️  크롤링 실패, 기존 데이터로 계속 진행합니다.")

        self.report.total_elapsed_sec = time.time() - total_start
        self.report.completed_at = datetime.now().isoformat()
        self.report.save(self.config.report_path)
        self.report.print_summary()

    # ============================================================
    # 1단계: 크롤링
    # ============================================================

    def _stage_crawl(self) -> StageResult:
        """키워드 생성 + 다중 소스 크롤링"""
        from ingestion.keyword_generator import KeywordGenerator
        from ingestion.multi_source_crawler import MultiSourceCrawler

        start = time.time()

        # 키워드 생성
        gen = KeywordGenerator(
            languages=self.config.languages,
            max_keywords=200,
        )
        keywords = gen.get_flat_keywords(max_count=50)

        print(f"  🔑 {len(keywords)}개 키워드 생성됨")
        for i, kw in enumerate(keywords[:10], 1):
            print(f"      {i}. {kw}")
        if len(keywords) > 10:
            print(f"      ... 외 {len(keywords) - 10}개")

        if self.config.dry_run:
            return StageResult(
                stage="crawl",
                success=True,
                count=0,
                details={"keywords": len(keywords), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        # ── 멀티프로세스 모드 ──
        if self.config.use_multiprocess:
            return self._crawl_multiprocess(keywords, start)

        # ── 기본 모드 (SingleProcess / Async) ──
        crawler = MultiSourceCrawler(
            sources=self.config.sources,
            max_results=self.config.crawl_target,
            max_workers=self.config.crawl_workers,
            get_full_info=self.config.crawl_full_info,
            min_duration_sec=self.config.min_duration_sec,
            max_duration_sec=self.config.max_duration_sec,
            content_filter=self.config.content_filter,
            async_mode=self.config.use_async,
        )

        results, stats = crawler.crawl(keywords)

        # CSV 저장
        csv_path = Path(self.config.urls_csv)
        crawler.save_csv(results, csv_path, overwrite=True)

        # DB 저장
        saved = crawler.save_to_db(results, self.config.db_path)

        self.report.total_crawled = len(results)

        return StageResult(
            stage="crawl",
            success=len(results) > 0,
            count=len(results),
            errors=stats.total_errors,
            elapsed_sec=time.time() - start,
            details={
                "keywords_used": len(keywords),
                "db_saved": saved,
                "duplicates": stats.total_duplicates,
                "filtered": stats.total_filtered,
                "by_source": stats.by_source,
                "mode": "async" if self.config.use_async else "sync",
            },
        )

    def _crawl_multiprocess(self, keywords: List[str], start: float) -> StageResult:
        """멀티프로세스 크롤링 (Task 2.1 통합)"""
        from workers.crawl_worker import run_workers
        from queue.task_queue import CrawlTaskQueue

        print(f"  🏭 멀티프로세스 모드 (워커: {self.config.crawl_workers}개)")

        try:
            stats = run_workers(
                num_workers=self.config.crawl_workers,
                keywords=keywords,
                source=self.config.sources[0] if self.config.sources else "youtube",
                max_results_per_task=self.config.crawl_target // max(len(keywords), 1),
            )

            # 결과 수집
            queue = CrawlTaskQueue()
            all_results = queue.get_all_results()

            total_count = sum(len(v) for v in all_results.values()) if isinstance(all_results, dict) else 0
            self.report.total_crawled = total_count

            return StageResult(
                stage="crawl",
                success=total_count > 0,
                count=total_count,
                errors=stats.get("total_failed", 0) if isinstance(stats, dict) else 0,
                elapsed_sec=time.time() - start,
                details={
                    "keywords_used": len(keywords),
                    "mode": "multiprocess",
                    "workers": self.config.crawl_workers,
                    "completed": stats.get("total_completed", 0) if isinstance(stats, dict) else 0,
                },
            )
        except Exception as e:
            logger.error(f"멀티프로세스 크롤링 실패, 기본 모드로 폴백: {e}")
            self.config.use_multiprocess = False
            return self._stage_crawl()

    # ============================================================
    # 2단계: 다운로드
    # ============================================================

    def _stage_download(self) -> StageResult:
        """병렬 다운로드"""
        import csv as csv_module

        start = time.time()
        csv_path = Path(self.config.urls_csv)
        output_dir = Path(self.config.raw_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV에서 URL 로드
        videos = []
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    if row.get("url") and row.get("video_id"):
                        videos.append({
                            "video_id": row["video_id"],
                            "url": row["url"],
                            "title": row.get("title", ""),
                        })

        if not videos:
            # DB에서 로드 시도
            videos = self._load_videos_from_db()

        if not videos:
            return StageResult(
                stage="download", success=False, errors=1,
                details={"error": "다운로드할 URL이 없습니다"},
                elapsed_sec=time.time() - start,
            )

        # 이미 다운로드된 파일 확인 (resume)
        if self.config.resume:
            existing = set(p.stem for p in output_dir.glob("*.mp4"))
            before = len(videos)
            videos = [v for v in videos if v["video_id"] not in existing]
            skipped = before - len(videos)
            if skipped > 0:
                print(f"  ⏭️ 이미 다운로드됨: {skipped}개 스킵")

        # 목표 수만큼만 다운로드
        videos = videos[:self.config.target_count]

        print(f"  📦 다운로드 대상: {len(videos)}개")

        if self.config.dry_run:
            return StageResult(
                stage="download",
                success=True,
                count=0,
                details={"target": len(videos), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        # parallel_download 모듈 사용
        from parallel_download import parallel_download, save_results_to_db

        results = parallel_download(
            videos=videos,
            output_dir=output_dir,
            num_workers=self.config.download_workers,
            timeout=self.config.download_timeout,
        )

        success_count = sum(1 for r in results if r.success and not r.skipped)
        skip_count = sum(1 for r in results if r.skipped)
        fail_count = sum(1 for r in results if not r.success)

        # DB 업데이트
        db_saved = save_results_to_db(results, videos)

        self.report.total_downloaded = success_count + skip_count

        return StageResult(
            stage="download",
            success=success_count > 0,
            count=success_count,
            errors=fail_count,
            elapsed_sec=time.time() - start,
            details={
                "new_downloads": success_count,
                "skipped": skip_count,
                "failed": fail_count,
                "db_saved": db_saved,
            },
        )

    # ============================================================
    # 3단계: 객체 검출 & Episode 생성
    # ============================================================

    def _stage_detect(self) -> StageResult:
        """객체 검출 및 Episode 생성 (GPU 3-Stream 통합)"""
        from extraction.detect_to_episodes import run as run_detect

        start = time.time()
        db_path = Path(self.config.db_path)
        output_dir = Path(self.config.episodes_dir)

        # 이미 생성된 에피소드 확인
        existing_episodes = set()
        if output_dir.exists():
            existing_episodes = {p.stem.replace("_episode", "") for p in output_dir.glob("*.npz")}

        print(f"  📂 기존 에피소드: {len(existing_episodes)}개")

        if self.config.dry_run:
            return StageResult(
                stage="detect",
                success=True,
                count=0,
                details={"existing": len(existing_episodes), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        # ── GPU 3-Stream 병렬 처리 ──
        gpu_info = {}
        if self.config.use_gpu_streams:
            try:
                from gpu.stream_manager import GPU3StreamManager
                stream_mgr = GPU3StreamManager()
                batch_size = stream_mgr.auto_adjust_batch_size()
                vram = stream_mgr.get_vram_usage()
                gpu_info = {
                    "gpu_streams": True,
                    "batch_size": batch_size,
                    "vram_allocated_gb": vram.get("allocated", 0),
                }
                print(f"  🎮 GPU 3-Stream 활성화 (배치: {batch_size}, VRAM: {vram.get('allocated', 0):.1f}GB)")
            except Exception as e:
                logger.warning(f"GPU 3-Stream 초기화 실패, 기본 모드 사용: {e}")
                gpu_info = {"gpu_streams": False, "reason": str(e)}

        # detect_to_episodes 실행
        try:
            run_detect(
                db_path=db_path,
                output_dir=output_dir,
                limit=self.config.target_count,
                use_redis=False,
                output_fps=self.config.detect_fps,
                device=self.config.detect_device,
            )
        except Exception as e:
            logger.error(f"검출 실패: {e}")

        # 새로 생성된 에피소드 확인
        new_episodes = set()
        if output_dir.exists():
            new_episodes = {p.stem for p in output_dir.glob("*.npz")}
        
        new_count = len(new_episodes) - len(existing_episodes)
        total_episodes = len(new_episodes)

        self.report.total_episodes = total_episodes

        return StageResult(
            stage="detect",
            success=total_episodes > 0,
            count=new_count,
            elapsed_sec=time.time() - start,
            details={
                "total_episodes": total_episodes,
                "new_episodes": new_count,
                "device": self.config.detect_device or "auto",
                **gpu_info,
            },
        )

    # ============================================================
    # 3.5단계: 품질 평가 (Task 2.3 통합)
    # ============================================================

    def _stage_quality(self) -> StageResult:
        """에피소드 품질 평가 및 필터링"""
        from quality.evaluator import RobotArmQualityEvaluator, QualityStats, QualityConfig
        import numpy as np

        start = time.time()
        episodes_dir = Path(self.config.episodes_dir)

        if not episodes_dir.exists():
            return StageResult(
                stage="quality", success=False, errors=1,
                details={"error": "에피소드 디렉토리가 없습니다"},
                elapsed_sec=time.time() - start,
            )

        npz_files = list(episodes_dir.glob("*.npz"))
        if not npz_files:
            return StageResult(
                stage="quality", success=True, count=0,
                details={"message": "평가할 에피소드 없음"},
                elapsed_sec=time.time() - start,
            )

        print(f"  🔍 품질 평가 대상: {len(npz_files)}개 에피소드")

        if not self.config.quality_filter:
            print(f"  ⏭️ 품질 필터링 비활성화")
            return StageResult(
                stage="quality", success=True, count=len(npz_files),
                details={"skipped": True, "filter_disabled": True},
                elapsed_sec=time.time() - start,
            )

        if self.config.dry_run:
            return StageResult(
                stage="quality", success=True, count=0,
                details={"target": len(npz_files), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        # 품질 평가
        config = QualityConfig(pass_threshold=self.config.quality_threshold)
        evaluator = RobotArmQualityEvaluator(config=config)
        stats = QualityStats()

        passed_count = 0
        failed_count = 0
        errors = 0

        for npz_path in npz_files:
            try:
                data = np.load(str(npz_path), allow_pickle=True)

                # 시퀀스 데이터 추출
                sequence = {
                    "body": data.get("body_landmarks", data.get("poses", np.array([]))),
                    "left_hand": data.get("left_hand_landmarks", np.array([])),
                    "right_hand": data.get("right_hand_landmarks", np.array([])),
                }

                video_id = npz_path.stem.replace("_episode", "")
                result = evaluator.evaluate(sequence, video_id=video_id)
                stats.record(result)

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    # 실패한 에피소드 이동 (삭제 대신 rejected 폴더로)
                    rejected_dir = episodes_dir / "rejected"
                    rejected_dir.mkdir(exist_ok=True)
                    npz_path.rename(rejected_dir / npz_path.name)

            except Exception as e:
                errors += 1
                logger.warning(f"품질 평가 실패: {npz_path.name} - {e}")

        # 보고서 출력
        stats.print_report()

        return StageResult(
            stage="quality",
            success=passed_count > 0,
            count=passed_count,
            errors=errors,
            elapsed_sec=time.time() - start,
            details={
                "total_evaluated": passed_count + failed_count,
                "passed": passed_count,
                "rejected": failed_count,
                "pass_rate": f"{stats.pass_rate:.1f}%" if hasattr(stats, 'pass_rate') else "N/A",
                "threshold": self.config.quality_threshold,
                "grades": stats.grades if hasattr(stats, 'grades') else {},
            },
        )

    # ============================================================
    # 4단계: S3 업로드
    # ============================================================

    def _stage_upload(self) -> StageResult:
        """S3 업로드"""
        start = time.time()
        episodes_dir = Path(self.config.episodes_dir)

        if not episodes_dir.exists():
            return StageResult(
                stage="upload", success=False, errors=1,
                details={"error": "에피소드 디렉토리가 없습니다"},
                elapsed_sec=time.time() - start,
            )

        npz_files = list(episodes_dir.glob("*.npz"))
        print(f"  📂 업로드 대상: {len(npz_files)}개 파일")

        if not npz_files:
            return StageResult(
                stage="upload", success=True, count=0,
                details={"message": "업로드할 파일 없음"},
                elapsed_sec=time.time() - start,
            )

        if self.config.dry_run:
            total_size = sum(f.stat().st_size for f in npz_files)
            return StageResult(
                stage="upload",
                success=True,
                count=0,
                details={
                    "files": len(npz_files),
                    "total_size_mb": total_size / (1024 ** 2),
                    "dry_run": True,
                },
                elapsed_sec=time.time() - start,
            )

        # upload_to_s3 모듈 사용
        try:
            from upload_to_s3 import get_s3_provider, get_bucket_name, upload_file

            provider = get_s3_provider()
            bucket = self.config.s3_bucket or get_bucket_name()

            uploaded = 0
            errors = 0

            for npz_file in npz_files:
                try:
                    result = upload_file(
                        provider=provider,
                        local_path=npz_file,
                        bucket=bucket,
                        prefix=self.config.s3_prefix,
                        data_type="episode",
                    )
                    if result.get("status") in ("uploaded", "completed"):
                        uploaded += 1
                    elif result.get("status") == "skipped":
                        uploaded += 1  # 이미 존재
                except Exception as e:
                    errors += 1
                    logger.error(f"업로드 실패: {npz_file.name} - {e}")

            self.report.total_uploaded = uploaded

            return StageResult(
                stage="upload",
                success=uploaded > 0,
                count=uploaded,
                errors=errors,
                elapsed_sec=time.time() - start,
                details={"bucket": bucket, "prefix": self.config.s3_prefix},
            )

        except Exception as e:
            return StageResult(
                stage="upload", success=False, errors=1,
                details={"error": str(e)},
                elapsed_sec=time.time() - start,
            )

    # ============================================================
    # 유틸리티
    # ============================================================

    def _load_videos_from_db(self) -> List[Dict]:
        """DB에서 discovered 상태 비디오 로드"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from models.database import Base, Video

            db_file = Path(self.config.db_path)
            if not db_file.is_absolute():
                db_file = project_root / db_file

            engine = create_engine(f"sqlite:///{db_file}")
            Session = sessionmaker(bind=engine)
            session = Session()

            videos = session.query(Video).filter(
                Video.status == "discovered",
                Video.url.isnot(None),
            ).limit(self.config.target_count).all()

            result = [
                {
                    "video_id": v.video_id,
                    "url": v.url,
                    "title": v.title or "",
                }
                for v in videos
            ]
            session.close()
            return result

        except Exception as e:
            logger.error(f"DB 로드 실패: {e}")
            return []


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="P-ADE 대량 수집 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예:
  # 전체 파이프라인 (500개 목표)
  python mass_collector.py --target 500

  # 크롤링만 실행
  python mass_collector.py --target 500 --stage crawl

  # 다운로드부터 재시작
  python mass_collector.py --target 500 --start-stage download

  # 특정 단계 구간 실행
  python mass_collector.py --start-stage download --end-stage detect

  # 드라이런
  python mass_collector.py --target 500 --dry-run

  # 커스텀 키워드
  python mass_collector.py --target 200 --keywords "robot arm,pick and place"

  # 소스 지정
  python mass_collector.py --target 500 --sources youtube,google_videos,vimeo
        """
    )

    parser.add_argument("--target", type=int, default=500, help="수집 목표 수 (기본: 500)")
    parser.add_argument("--stage", help="단일 단계 실행: crawl, download, detect, quality, upload")
    parser.add_argument("--start-stage", help="시작 단계")
    parser.add_argument("--end-stage", help="종료 단계")
    parser.add_argument("--keywords", help="커스텀 키워드 (콤마 구분)")
    parser.add_argument("--sources", default="youtube,google_videos",
                       help="소스 (기본: youtube,google_videos)")
    parser.add_argument("--languages", default="en,ko", help="키워드 언어 (기본: en,ko)")
    parser.add_argument("--crawl-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--download-timeout", type=int, default=600)
    parser.add_argument("--detect-fps", type=float, default=5.0)
    parser.add_argument("--detect-device", default=None, help="검출 디바이스 (예: cuda:0)")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--s3-prefix", default="episodes")
    parser.add_argument("--db", default="data/pade.db")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--dry-run", action="store_true", help="실제 실행 없이 계획만 출력")
    parser.add_argument("--no-resume", action="store_true", help="이전 진행 무시")
    parser.add_argument("--min-duration", type=int, default=30)
    parser.add_argument("--max-duration", type=int, default=1200)
    parser.add_argument("--multiprocess", action="store_true", help="멀티프로세스 크롤링 모드")
    parser.add_argument("--async", dest="use_async", action="store_true", help="비동기 크롤링 모드")
    parser.add_argument("--no-gpu-streams", action="store_true", help="GPU 3-Stream 비활성화")
    parser.add_argument("--no-quality-filter", action="store_true", help="품질 필터링 비활성화")
    parser.add_argument("--quality-threshold", type=float, default=60.0, help="품질 통과 점수 (기본: 60)")

    args = parser.parse_args()

    config = PipelineConfig(
        target_count=args.target,
        sources=[s.strip() for s in args.sources.split(",")],
        languages=[l.strip() for l in args.languages.split(",")],
        crawl_workers=args.crawl_workers,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        download_workers=args.download_workers,
        download_timeout=args.download_timeout,
        detect_fps=args.detect_fps,
        detect_device=args.detect_device,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        db_path=args.db,
        raw_dir=args.output_dir,
        episodes_dir=args.episodes_dir,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        use_multiprocess=args.multiprocess,
        use_async=args.use_async,
        use_gpu_streams=not args.no_gpu_streams,
        quality_filter=not args.no_quality_filter,
        quality_threshold=args.quality_threshold,
    )

    collector = MassCollector(config)

    if args.stage:
        # 단일 단계 실행
        collector.run(start_stage=args.stage, end_stage=args.stage)
    elif args.start_stage or args.end_stage:
        collector.run(start_stage=args.start_stage, end_stage=args.end_stage)
    else:
        # 전체 파이프라인 실행
        collector.run()


if __name__ == "__main__":
    main()
