#!/usr/bin/env python
"""
P-ADE 대량 수집 오케스트레이터

500개 이상의 로봇팔 영상을 자동으로 수집하는 end-to-end 파이프라인입니다.

파이프라인 단계:
  1. 키워드 생성 → 다중 소스 크롤링 (URL 수집)
  2. 병렬 다운로드 (비디오 파일 저장)
  3. 객체 검출 & Episode 생성
  4. 모방학습 데이터 생성 (포즈 추출 → State-Action 인코딩)
  5. 품질 평가 및 필터링
  6. S3 클라우드 업로드
  7. 통계 리포트 출력

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
    target_count: int = 5000
    
    # 크롤링 설정
    sources: List[str] = field(default_factory=lambda: ["youtube", "google_videos", "vimeo", "bilibili"])
    languages: List[str] = field(default_factory=lambda: ["en", "ko", "ja", "zh", "de"])
    crawl_workers: int = 16
    crawl_full_info: bool = False
    min_duration_sec: int = 30
    max_duration_sec: int = 1200
    content_filter: bool = True
    max_keywords: int = 500           # 크롤링 키워드 수
    use_multiprocess: bool = True     # 멀티프로세스 크롤링 모드
    use_async: bool = True            # 비동기 크롤링 모드 (기본: True, Task A2-2)

    # 다운로드 설정
    download_workers: int = 12
    download_timeout: int = 300
    download_quality: str = "480p"
    
    # 검출 설정
    detect_fps: float = 5.0
    detect_device: Optional[str] = None  # None = auto-detect
    detect_batch_size: int = 50
    use_gpu_streams: bool = True  # GPU 3-stream 병렬 처리
    
    # 품질 평가 설정
    quality_filter: bool = True  # 품질 필터링 활성화
    quality_threshold: float = 50.0  # 통과 점수 (60 → 50 완화, 목표 통과율 65%)

    # 통합 처리 설정 (1-Pass Detect+IL)
    unified_processing: bool = True   # True=1-Pass 통합, False=기존 2-Pass
    num_gpu_streams: int = 6          # GPU 스트림 수 (dual-GPU: 3+3)

    # 모방학습 데이터 생성 설정 (unified_processing=False 시 사용)
    build_il: bool = True  # 모방학습 데이터 생성 활성화
    il_fps: float = 5.0  # 추출 FPS
    il_max_frames: Optional[int] = None  # 비디오당 최대 프레임
    
    # 업로드 설정
    s3_bucket: str = ""
    s3_prefix: str = "episodes"
    upload_workers: int = 8
    cleanup_after_upload: bool = True  # S3 업로드 후 로컬 파일 삭제
    
    # 파이프라인 실행 식별
    run_id: str = ""                   # 실행 ID (비어있으면 자동 생성)
    
    # 경로
    db_path: str = "data/pade.db"
    raw_dir: str = "data/raw"
    episodes_dir: str = "data/episodes"
    urls_csv: str = "data/urls_mass.csv"
    report_path: str = "data/collection_report.json"
    
    # 기타
    dry_run: bool = False
    resume: bool = True  # 이전 진행 이어받기
    custom_keywords: Optional[List[str]] = None  # CLI에서 직접 지정한 키워드
    
    @property
    def crawl_multiplier(self) -> float:
        """목표 대비 크롤링 초과 수집 배수 (필터링 감안)"""
        return 4.0
    
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
    total_il_episodes: int = 0
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
        print(f"    IL 데이터:  {self.total_il_episodes}개")
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

    STAGES = ["crawl", "download", "process", "quality", "upload", "cleanup"]

    def __init__(self, config: PipelineConfig, on_stage_start=None, on_stage_complete=None, on_log=None):
        """
        Args:
            config: 파이프라인 설정
            on_stage_start: 스테이지 시작 콜백 fn(stage_name)
            on_stage_complete: 스테이지 완료 콜백 fn(stage_name, result: StageResult)
            on_log: 로그 콜백 fn(message: str)
        """
        self.config = config
        self.report = PipelineReport(
            started_at=datetime.now().isoformat(),
            config=asdict(config),
        )
        self._on_stage_start = on_stage_start
        self._on_stage_complete = on_stage_complete
        self._on_log = on_log

    def _notify_stage_start(self, stage_name: str):
        """스테이지 시작 알림"""
        if self._on_stage_start:
            try:
                self._on_stage_start(stage_name)
            except Exception:
                pass

    def _notify_stage_complete(self, stage_name: str, result: 'StageResult'):
        """스테이지 완료 알림"""
        if self._on_stage_complete:
            try:
                self._on_stage_complete(stage_name, result)
            except Exception:
                pass

    def _notify_log(self, message: str):
        """로그 알림"""
        if self._on_log:
            try:
                self._on_log(message)
            except Exception:
                pass

    def _publish_progress(self, stage: str, status: str, current: int, total: int, count: int = 0):
        """Redis에 진행률 발행 (D1-2: SSE 스트리밍용)"""
        try:
            import redis as _redis
            r = _redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_timeout=2,
            )
            percent = (current / total * 100) if total > 0 else 0
            r.publish("pade:progress", json.dumps({
                "stage": stage,
                "status": status,
                "current": current,
                "total": total,
                "percent": round(percent, 1),
                "count": count,
                "timestamp": datetime.now().isoformat(),
            }))
        except Exception:
            pass  # Redis 미연결 시 무시

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

        header = (
            f"\n🚀{'='*68}\n"
            f"   P-ADE 대량 수집 파이프라인\n"
            f"{'='*70}\n"
            f"   목표: {self.config.target_count}개 영상\n"
            f"   소스: {', '.join(self.config.sources)}\n"
            f"   단계: {' → '.join(stages)}\n"
            f"   드라이런: {'예' if self.config.dry_run else '아니오'}\n"
            f"{'='*70}"
        )
        print(header)
        self._notify_log(f"파이프라인 시작: {' → '.join(stages)} (목표: {self.config.target_count})")

        # 파이프라인 시작 시 이전 실행 잔여파일 정리 (C3-2)
        try:
            from storage.disk_policy import DiskPolicy
            disk = DiskPolicy(
                raw_dir=self.config.raw_dir,
                episodes_dir=self.config.episodes_dir,
            )
            count, freed = disk.cleanup_old_runs(days=7)
            if count > 0:
                print(f"  🧹 이전 실행 파일 정리: {count}개 삭제, {freed:.2f}GB 확보")
        except Exception as e:
            logger.warning(f"이전 실행 파일 정리 실패 (무시): {e}")

        for stage_name in stages:
            print(f"\n{'─'*70}")
            print(f"📌 단계: {stage_name.upper()}")
            print(f"{'─'*70}")

            handler = getattr(self, f"_stage_{stage_name}", None)
            if not handler:
                msg = f"알 수 없는 단계: {stage_name}"
                logger.error(msg)
                self._notify_log(f"[ERROR] {msg}")
                continue

            # 스테이지 시작 알림
            self._notify_stage_start(stage_name)
            self._notify_log(f"[INFO] ▶ {stage_name.upper()} 단계 시작...")

            # Redis 진행률 발행 (D1-2)
            self._publish_progress(stage_name, "started", 0, len(stages))

            try:
                result = handler()
                self.report.stages.append(asdict(result))
                print(f"  {result.summary()}")

                # Redis 진행률 발행 (D1-2)
                stage_idx = stages.index(stage_name) + 1
                self._publish_progress(
                    stage_name, "completed" if result.success else "failed",
                    stage_idx, len(stages), result.count,
                )

                # 스테이지 완료 알림
                self._notify_stage_complete(stage_name, result)
                self._notify_log(f"[{'SUCCESS' if result.success else 'WARN'}] {result.summary()}")

                # 실패 시에도 다음 단계로 계속 진행 (기존 데이터 활용)
                if not result.success:
                    msg = f"⚠️ {stage_name} 단계 실패 — 기존 데이터로 다음 단계를 계속 진행합니다."
                    logger.warning(msg)
                    self._notify_log(f"[WARN] {msg}")

            except Exception as e:
                logger.error(f"단계 실패 [{stage_name}]: {e}")
                result = StageResult(
                    stage=stage_name, success=False, errors=1,
                    details={"error": str(e)},
                )
                self.report.stages.append(asdict(result))
                print(f"  {result.summary()}")

                self._notify_stage_complete(stage_name, result)
                self._notify_log(f"[ERROR] {stage_name} 단계 예외: {e}")

                # 크롤링/다운로드 실패 시 기존 데이터로 계속 진행
                if stage_name in ("crawl", "download"):
                    print(f"  ⚠️  {stage_name} 실패, 기존 데이터로 계속 진행합니다.")
                else:
                    # 그 외 단계도 경고만 남기고 계속 진행
                    msg = f"⚠️ {stage_name} 단계 예외 — 기존 데이터로 다음 단계를 계속 진행합니다."
                    logger.warning(msg)
                    self._notify_log(f"[WARN] {msg}")

        self.report.total_elapsed_sec = time.time() - total_start
        self.report.completed_at = datetime.now().isoformat()
        self.report.save(self.config.report_path)
        self.report.print_summary()
        self._notify_log(f"[INFO] 파이프라인 완료 (소요: {self.report.total_elapsed_sec:.1f}초)")

    # ============================================================
    # 1단계: 크롤링
    # ============================================================

    def _stage_crawl(self) -> StageResult:
        """키워드 생성 + 다중 소스 크롤링"""
        from ingestion.keyword_generator import KeywordGenerator
        from ingestion.multi_source_crawler import MultiSourceCrawler

        start = time.time()

        # 키워드 생성
        if self.config.custom_keywords:
            keywords = self.config.custom_keywords
        else:
            gen = KeywordGenerator(
                languages=self.config.languages,
                max_keywords=self.config.max_keywords,
            )
            # 카테시안 조합 풀에서 상위 키워드 추출
            cartesian = gen.generate_cartesian_all()
            flat = gen.get_flat_keywords(max_count=self.config.max_keywords)
            # 카테시안 + 기존을 합쳐서 중복 제거 후 상위 max_keywords
            seen = set()
            keywords = []
            for kw in flat + cartesian:
                kw_lower = kw.lower().strip()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    keywords.append(kw)
            keywords = keywords[:self.config.max_keywords]

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

        # GlobalVideoRegistry 필터: 이미 수집된 영상 제거
        try:
            from cache.video_registry import get_registry
            registry = get_registry(self.config.db_path)
            result_dicts = [
                {"video_id": r.video_id, "url": r.url}
                for r in results
            ]
            new_dicts = registry.filter_new_only(result_dicts)
            new_ids = {d["video_id"] for d in new_dicts}
            before = len(results)
            results = [r for r in results if r.video_id in new_ids]
            registry_blocked = before - len(results)
            if registry_blocked > 0:
                logger.info(f"🚫 Registry 중복 차단: {registry_blocked}개, 신규: {len(results)}개")
        except Exception as e:
            logger.warning(f"Registry 필터 실패 (무시): {e}")
            registry_blocked = 0

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
                "registry_blocked": registry_blocked,
                "by_source": stats.by_source,
                "mode": "async" if self.config.use_async else "sync",
            },
        )

    def _crawl_multiprocess(self, keywords: List[str], start: float) -> StageResult:
        """멀티프로세스 크롤링 (Task 2.1 통합)"""
        from workers.crawl_worker import run_workers
        from task_queue.task_queue import CrawlTaskQueue

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

        # GlobalVideoRegistry 체크: 이미 수집된 video_id 스킵
        try:
            from cache.video_registry import get_registry
            registry = get_registry(self.config.db_path)
            before = len(videos)
            videos = [v for v in videos if not registry.is_collected(v["video_id"])]
            registry_skipped = before - len(videos)
            if registry_skipped > 0:
                logger.info(f"Already collected: {registry_skipped}, New: {len(videos)}")
                print(f"  🚫 Registry 중복: {registry_skipped}개 스킵")
        except Exception as e:
            logger.warning(f"Registry 체크 실패 (무시): {e}")

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

        # 디스크 공간 확인 (C3-2: 다운로드 전 ensure_space)
        try:
            from storage.disk_policy import DiskPolicy
            disk = DiskPolicy(
                raw_dir=str(output_dir),
                episodes_dir=self.config.episodes_dir,
            )
            if not disk.ensure_space():
                return StageResult(
                    stage="download", success=False, errors=1,
                    details={"error": "디스크 공간 부족 (최소 100GB 필요)"},
                    elapsed_sec=time.time() - start,
                )
        except Exception as e:
            logger.warning(f"디스크 공간 확인 실패 (무시): {e}")

        # parallel_download 모듈 사용
        from scripts.pipeline.parallel_download import parallel_download, save_results_to_db

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
    # 3단계: 통합 처리 (1-Pass Detect + IL) — Task B1-1/B1-2
    # ============================================================

    def _stage_process(self) -> StageResult:
        """
        1-Pass 통합 처리: YOLO 객체 검출 + MediaPipe 포즈 추출 + State-Action 인코딩
        비디오 1회 디코딩으로 Detect+IL을 동시에 처리 (처리 시간 40% 단축 목표)
        """
        start = time.time()
        raw_dir = Path(self.config.raw_dir)
        episodes_dir = Path(self.config.episodes_dir)
        episodes_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(raw_dir.glob("*.mp4"))
        if not video_files:
            return StageResult(
                stage="process", success=False, errors=1,
                details={"error": "처리할 비디오 없음"},
                elapsed_sec=time.time() - start,
            )

        # 이미 처리된 영상 스킵
        existing = {p.stem.replace("_episode", "") for p in episodes_dir.glob("*.npz")}
        pending = [str(v) for v in video_files if v.stem not in existing]

        print(f"  📂 비디오: {len(video_files)}개 (기처리: {len(existing)}개, 대상: {len(pending)}개)")

        if not pending:
            return StageResult(
                stage="process", success=True, count=len(existing),
                details={"message": "모든 비디오가 이미 처리됨", "skipped": len(existing)},
                elapsed_sec=time.time() - start,
            )

        if self.config.dry_run:
            return StageResult(
                stage="process", success=True, count=0,
                details={"target": len(pending), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        success_count = 0
        fail_count = 0

        if self.config.unified_processing:
            # ── 1-Pass 통합 처리 (UnifiedVideoProcessor + GPU 6-Stream) ──
            try:
                from gpu.stream_manager import GPU3StreamManager, StreamConfig
                from gpu.unified_processor import UnifiedVideoProcessor

                stream_cfg = StreamConfig(
                    num_streams=self.config.num_gpu_streams,
                    streams_per_gpu=self.config.num_gpu_streams // 2,
                )
                stream_mgr = GPU3StreamManager(config=stream_cfg)

                unified_proc = UnifiedVideoProcessor(
                    output_fps=self.config.il_fps,
                    device=self.config.detect_device or "cuda:0",
                    max_frames=self.config.il_max_frames,
                )

                def _unified_processor_fn(video_path: str):
                    video_id = Path(video_path).stem
                    out_path = str(episodes_dir / f"{video_id}_episode.npz")
                    return unified_proc.process(video_path, out_path)

                results = stream_mgr.process_batch(
                    pending[:self.config.target_count],
                    processor=_unified_processor_fn,
                    output_dir=str(episodes_dir),
                )

                stream_mgr.print_stats()

                for r in results:
                    if r and r.get("success"):
                        success_count += 1
                        status = r.get("status", "success")
                        if status != "skipped":
                            print(f"  ✅ {r.get('video_id', '?')}: {r.get('frames', 0)}f "
                                  f"S:{r.get('state_dim', '?')} A:{r.get('action_dim', '?')}")
                    else:
                        fail_count += 1
                        print(f"  ❌ {r.get('video_id', '?')}: {r.get('error', 'unknown')}")

                self.report.total_episodes = len(existing) + success_count
                self.report.total_il_episodes = len(existing) + success_count

            except Exception as e:
                logger.error(f"통합 처리 실패, 폴백: {e}")
                # 폴백: 기존 방식으로 실행
                self.config.unified_processing = False
                return self._stage_process()

        else:
            # ── 폴백: 기존 2-Pass (detect → build_il) ──
            try:
                r1 = self._stage_detect()
                r2 = self._stage_build_il()
                success_count = r1.count + r2.count
                fail_count = r1.errors + r2.errors
            except Exception as e:
                logger.error(f"폴백 처리 실패: {e}")
                fail_count += 1

        return StageResult(
            stage="process",
            success=success_count > 0,
            count=success_count,
            errors=fail_count,
            elapsed_sec=time.time() - start,
            details={
                "mode": "unified_1pass" if self.config.unified_processing else "legacy_2pass",
                "new_processed": success_count,
                "previously_done": len(existing),
                "failed": fail_count,
                "fps": self.config.il_fps,
                "gpu_streams": self.config.num_gpu_streams,
            },
        )

    def _stage_cleanup(self) -> StageResult:
        """정리 단계: 처리 완료된 원본 MP4 삭제, 품질 탈락 NPZ 삭제, 통계 출력"""
        start = time.time()
        total_deleted = 0
        total_freed = 0.0

        try:
            from storage.disk_policy import DiskPolicy
            disk = DiskPolicy(
                raw_dir=self.config.raw_dir,
                episodes_dir=self.config.episodes_dir,
            )

            # 처리 완료된 원본 MP4 삭제 (에피소드가 있는 영상의 원본)
            episodes_dir = Path(self.config.episodes_dir)
            if episodes_dir.exists():
                processed_ids = [
                    p.stem.replace("_episode", "")
                    for p in episodes_dir.glob("*.npz")
                ]
                if processed_ids:
                    count, freed = disk.cleanup_raw_videos(video_ids=processed_ids)
                    total_deleted += count
                    total_freed += freed

            # 품질 탈락 NPZ 삭제
            count, freed = disk.cleanup_rejected()
            total_deleted += count
            total_freed += freed

            usage = disk.get_disk_usage()
            print(f"  🧹 정리 완료: {total_deleted}개 삭제, {total_freed:.2f}GB 확보")
            print(f"  💾 디스크 여유: {usage['free_gb']:.1f}GB ({100 - usage['usage_percent']:.1f}%)")

        except Exception as e:
            logger.warning(f"정리 단계 오류 (무시): {e}")

        return StageResult(
            stage="cleanup", success=True, count=total_deleted,
            details={
                "deleted_files": total_deleted,
                "freed_gb": round(total_freed, 2),
            },
            elapsed_sec=time.time() - start,
        )

    # ============================================================
    # (레거시) 3단계: 객체 검출 & Episode 생성
    # ============================================================

    def _stage_detect(self) -> StageResult:
        """객체 검출 및 Episode 생성 (GPU 3-Stream 병렬 처리 통합)"""
        from extraction.detect_to_episodes import run as run_detect

        start = time.time()
        db_path = Path(self.config.db_path)
        output_dir = Path(self.config.episodes_dir)
        raw_dir = Path(self.config.raw_dir)

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

        gpu_info = {}
        gpu_batch_used = False

        # ── GPU 3-Stream 병렬 처리 시도 ──
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

                # 미처리 비디오 수집
                video_files = sorted(raw_dir.glob("*.mp4"))
                pending = [str(v) for v in video_files if v.stem not in existing_episodes]

                if pending:
                    print(f"  🔍 GPU 3-Stream으로 {len(pending)}개 비디오 검출 처리...")
                    detect_processor = stream_mgr.make_detect_processor(
                        output_fps=self.config.detect_fps,
                        device=self.config.detect_device,
                    )
                    results = stream_mgr.process_batch(pending[:self.config.target_count], detect_processor)

                    # 결과를 npz로 저장
                    import numpy as np
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for r in results:
                        if r.get("success") and r.get("detections"):
                            from extraction.detect_to_episodes import _save_episode_npz
                            out_path = output_dir / f"{r['video_id']}_episode.npz"
                            metadata = {
                                "video_id": r["video_id"],
                                "source_path": r["video_path"],
                                "num_frames": r["frames"],
                                "output_fps": self.config.detect_fps,
                            }
                            _save_episode_npz(out_path, r["detections"], metadata)

                    gpu_batch_used = True
                    stream_mgr.print_stats()
                    gpu_info["processed_by_gpu"] = sum(1 for r in results if r.get("success"))
                    gpu_info["failed_by_gpu"] = sum(1 for r in results if not r.get("success"))
            except Exception as e:
                logger.warning(f"GPU 3-Stream 검출 실패, 기본 모드로 폴백: {e}")
                gpu_info = {"gpu_streams": False, "reason": str(e)}

        # ── 폴백: 기본 순차 detect ──
        if not gpu_batch_used:
            try:
                run_detect(
                    db_path=db_path,
                    output_dir=output_dir,
                    limit=self.config.target_count,
                    use_redis=False,
                    output_fps=self.config.detect_fps,
                    device=self.config.detect_device,
                    use_gpu_streams=False,  # GPU 3-Stream은 이미 위에서 시도함
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
    # 3.5단계: 모방학습 데이터 생성 (build_imitation_data)
    # ============================================================

    def _stage_build_il(self) -> StageResult:
        """비디오 → 포즈 추출 → State-Action 인코딩 → .npz 저장 (GPU 3-Stream 병렬 처리)"""
        from scripts.pipeline.build_imitation_data import process_single_video
        import numpy as np

        start = time.time()
        raw_dir = Path(self.config.raw_dir)
        episodes_dir = Path(self.config.episodes_dir)
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # 다운로드된 비디오 목록
        videos = sorted(raw_dir.glob("*.mp4"))
        if not videos:
            return StageResult(
                stage="build_il", success=False, errors=1,
                details={"error": "모방학습 데이터로 변환할 비디오가 없습니다"},
                elapsed_sec=time.time() - start,
            )

        # 이미 IL 데이터가 있는 비디오 스킵
        existing_il = set()
        for f in episodes_dir.glob("*_episode.npz"):
            try:
                d = np.load(str(f), allow_pickle=True)
                if "states" in d and "actions" in d:
                    existing_il.add(f.stem.replace("_episode", ""))
            except Exception:
                pass

        pending = [v for v in videos if v.stem not in existing_il]
        print(f"  📂 비디오: {len(videos)}개 (기존 IL: {len(existing_il)}개, 대상: {len(pending)}개)")

        if not pending:
            return StageResult(
                stage="build_il", success=True, count=len(existing_il),
                details={"message": "모든 비디오의 IL 데이터가 이미 존재합니다", "skipped": len(existing_il)},
                elapsed_sec=time.time() - start,
            )

        if self.config.dry_run:
            return StageResult(
                stage="build_il", success=True, count=0,
                details={"target": len(pending), "dry_run": True},
                elapsed_sec=time.time() - start,
            )

        success_count = 0
        fail_count = 0
        gpu_used = False
        gpu_info = {}

        # ── GPU 3-Stream 병렬 처리 시도 ──
        if self.config.use_gpu_streams:
            try:
                from gpu.stream_manager import GPU3StreamManager
                stream_mgr = GPU3StreamManager()
                batch_size = stream_mgr.auto_adjust_batch_size()
                vram = stream_mgr.get_vram_usage()
                print(f"  🎮 GPU 3-Stream IL 생성 (배치: {batch_size}, VRAM: {vram.get('allocated', 0):.1f}GB)")

                il_processor = stream_mgr.make_il_processor(
                    output_fps=self.config.il_fps,
                    max_frames=self.config.il_max_frames,
                    output_dir=str(episodes_dir),
                )
                pending_paths = [str(v) for v in pending]
                results = stream_mgr.process_batch(pending_paths, il_processor)

                for r in results:
                    if r and r.get("success"):
                        if r.get("status") == "skipped":
                            print(f"  ⏭️  {r.get('video_id', '?')}: {r.get('msg', 'skipped')}")
                        else:
                            print(f"  ✅ {r.get('video_id', '?')}: {r.get('frames', 0)}f "
                                  f"S:{r.get('state_dim', '?')} A:{r.get('action_dim', '?')}")
                        success_count += 1
                    else:
                        fail_count += 1
                        print(f"  ❌ {r.get('video_id', '?')}: {r.get('error', 'unknown')}")

                stream_mgr.print_stats()
                gpu_used = True
                gpu_info = {
                    "gpu_streams": True,
                    "batch_size": batch_size,
                    "vram_gb": vram.get("allocated", 0),
                    "peak_vram_gb": stream_mgr.stats.get("peak_vram_gb", 0),
                }
            except Exception as e:
                logger.warning(f"GPU 3-Stream IL 생성 실패, 순차 모드로 폴백: {e}")
                gpu_info = {"gpu_streams": False, "reason": str(e)}

        # ── 폴백: 순차 처리 ──
        if not gpu_used:
            total = len(pending)
            for i, video_path in enumerate(pending, 1):
                vid = video_path.stem
                print(f"  [{i}/{total}] {vid}...", end=" ", flush=True)

                task = (
                    str(video_path),
                    str(episodes_dir),
                    self.config.il_fps,
                    self.config.il_max_frames,
                    i,
                    total,
                )

                result = process_single_video(task)

                if result["status"] == "success":
                    success_count += 1
                    print(f"✅ {result['frames']}f S:{result['state_dim']} A:{result['action_dim']} ({result['time']}s)")
                elif result["status"] == "skipped":
                    success_count += 1
                    print(f"⏭️  {result['msg']}")
                else:
                    fail_count += 1
                    print(f"❌ {result.get('msg', 'unknown')}")

        total_il = len(existing_il) + success_count

        return StageResult(
            stage="build_il",
            success=success_count > 0 or len(existing_il) > 0,
            count=success_count,
            errors=fail_count,
            elapsed_sec=time.time() - start,
            details={
                "total_il_episodes": total_il,
                "new_built": success_count,
                "previously_built": len(existing_il),
                "failed": fail_count,
                "fps": self.config.il_fps,
                **gpu_info,
            },
        )

    # ============================================================
    # 4단계: 품질 평가 (Task 2.3 통합)
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

        # 품질 평가 (배치 벡터화)
        config = QualityConfig(pass_threshold=self.config.quality_threshold)
        evaluator = RobotArmQualityEvaluator(config=config)
        stats = QualityStats()

        passed_count = 0
        failed_count = 0
        errors = 0

        # evaluate_batch() 호출 (DB 마킹으로 파일 이동 없이 처리)
        try:
            batch_results = evaluator.evaluate_batch(
                [str(p) for p in npz_files],
                db_path=self.config.db_path,
            )
        except Exception as e:
            logger.error(f"배치 평가 실패, 순차 평가로 폴백: {e}")
            batch_results = []
            for npz_path in npz_files:
                try:
                    batch_results.append(evaluator.evaluate_npz(str(npz_path)))
                except Exception as ex:
                    logger.warning(f"평가 실패: {npz_path.name} - {ex}")
                    errors += 1

        for result in batch_results:
            try:
                stats.record(result)
                npz_path = episodes_dir / f"{result.video_id}_episode.npz"

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    reason = result.fail_reason or "quality_check_failed"

                    # Registry에 rejected 등록 (재수집 방지)
                    try:
                        from cache.video_registry import get_registry
                        registry = get_registry(self.config.db_path)
                        registry.register_rejected(result.video_id, reason)
                    except Exception:
                        pass

                    # 파일 이동 대신 DB 마킹 (디스크 I/O 절약)
                    # evaluate_batch() 내부에서 이미 DB 마킹됨

            except Exception as e:
                errors += 1
                logger.warning(f"품질 결과 처리 실패: {result.video_id} - {e}")

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
            from scripts.pipeline.upload_to_s3 import get_s3_provider, get_bucket_name, upload_file

            provider = get_s3_provider()
            bucket = self.config.s3_bucket or get_bucket_name()

            uploaded = 0
            errors = 0

            # Registry 준비
            try:
                from cache.video_registry import get_registry
                registry = get_registry(self.config.db_path)
            except Exception:
                registry = None

            run_id = self.report.started_at

            for npz_file in npz_files:
                try:
                    result = upload_file(
                        provider=provider,
                        local_path=npz_file,
                        bucket=bucket,
                        prefix=self.config.s3_prefix,
                        data_type="episode",
                    )
                    status = result.get("status")
                    if status in ("uploaded", "completed"):
                        uploaded += 1
                        # S3 업로드 성공 시 Registry 등록
                        if registry:
                            video_id = npz_file.stem.replace("_episode", "")
                            s3_path = result.get("s3_path", "")
                            registry.register(video_id, "", run_id, s3_path)
                    elif status == "skipped":
                        uploaded += 1  # 이미 존재
                except Exception as e:
                    errors += 1
                    logger.error(f"업로드 실패: {npz_file.name} - {e}")

            self.report.total_uploaded = uploaded

            # S3 업로드 완료 후 로컬 파일 정리 (C3-2)
            cleanup_count = 0
            cleanup_freed = 0.0
            if uploaded > 0:
                try:
                    from storage.disk_policy import DiskPolicy
                    disk = DiskPolicy(
                        raw_dir=self.config.raw_dir,
                        episodes_dir=self.config.episodes_dir,
                    )
                    cleanup_count, cleanup_freed = disk.cleanup_after_upload(run_id=run_id)
                    if cleanup_count > 0:
                        print(f"  🧹 업로드 후 정리: {cleanup_count}개 삭제, {cleanup_freed:.2f}GB 확보")
                except Exception as e:
                    logger.warning(f"업로드 후 정리 실패 (무시): {e}")

            return StageResult(
                stage="upload",
                success=uploaded > 0,
                count=uploaded,
                errors=errors,
                elapsed_sec=time.time() - start,
                details={
                    "bucket": bucket,
                    "prefix": self.config.s3_prefix,
                    "cleanup_files": cleanup_count,
                    "cleanup_freed_gb": round(cleanup_freed, 2),
                },
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
    parser.add_argument("--stage", help="단일 단계 실행: crawl, download, detect, build_il, quality, upload")
    parser.add_argument("--start-stage", help="시작 단계")
    parser.add_argument("--end-stage", help="종료 단계")
    parser.add_argument("--keywords", help="커스텀 키워드 (콤마 구분)")
    parser.add_argument("--sources", default="youtube,google_videos,vimeo,bilibili",
                       help="소스 (기본: youtube,google_videos,vimeo,bilibili)")
    parser.add_argument("--languages", default="en,ko,ja,zh,de", help="키워드 언어 (기본: en,ko,ja,zh,de)")
    parser.add_argument("--crawl-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--download-timeout", type=int, default=300)
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
    parser.add_argument("--no-build-il", action="store_true", help="모방학습 데이터 생성 비활성화")
    parser.add_argument("--il-fps", type=float, default=5.0, help="IL 추출 FPS (기본: 5)")
    parser.add_argument("--il-max-frames", type=int, default=None, help="IL 비디오당 최대 프레임")
    parser.add_argument("--no-unified", action="store_true", help="1-Pass 통합 처리 비활성화 (기존 2-Pass 사용)")
    parser.add_argument("--gpu-streams", type=int, default=6, help="GPU 스트림 수 (기본: 6)")

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
        build_il=not args.no_build_il,
        il_fps=args.il_fps,
        il_max_frames=args.il_max_frames,
        unified_processing=not args.no_unified,
        num_gpu_streams=args.gpu_streams,
    )

    # 커스텀 키워드 적용
    if args.keywords:
        config.custom_keywords = [k.strip() for k in args.keywords.split(",")]

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
