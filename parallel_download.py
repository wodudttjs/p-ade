#!/usr/bin/env python
"""
병렬 다운로드 스크립트

MVP Phase 2 Week 5: Queue System
- 병렬 다운로드 (4 workers)
- 처리 속도 2배 개선

사용법:
    python parallel_download.py --query "2족보행" --limit 10 --workers 4
    python parallel_download.py --urls urls.txt --workers 4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from core.queue_manager import QueueManager, TaskPriority

# DB 저장용 import
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models.database import Video
    HAS_DB = True
except ImportError:
    HAS_DB = False

logger = setup_logger(__name__)


@dataclass
class DownloadResult:
    """다운로드 결과"""
    video_id: str
    url: str
    success: bool
    video_path: Optional[str] = None
    size_bytes: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None
    skipped: bool = False


def download_single(
    video_id: str,
    url: str,
    output_dir: Path,
    timeout: int = 300,
) -> DownloadResult:
    """단일 비디오 다운로드 (yt-dlp Python API 사용)"""
    import yt_dlp
    
    start_time = time.time()
    output_path = output_dir / f"{video_id}.mp4"
    
    # 이미 존재하면 스킵
    if output_path.exists():
        return DownloadResult(
            video_id=video_id,
            url=url,
            success=True,
            video_path=str(output_path),
            size_bytes=output_path.stat().st_size,
            skipped=True,
        )
    
    # Deno 런타임 경로를 PATH에 추가
    deno_dir = Path.home() / ".deno" / "bin"
    if deno_dir.exists() and str(deno_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(deno_dir) + os.pathsep + os.environ.get("PATH", "")
    
    try:
        ydl_opts = {
            "format": "best[height<=720]",
            "outtmpl": str(output_path),
            "no_playlist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": min(timeout, 60),
            "retries": 2,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if output_path.exists():
            duration = time.time() - start_time
            return DownloadResult(
                video_id=video_id,
                url=url,
                success=True,
                video_path=str(output_path),
                size_bytes=output_path.stat().st_size,
                duration_sec=duration,
            )
        else:
            return DownloadResult(
                video_id=video_id,
                url=url,
                success=False,
                error="Download completed but file not found",
            )
            
    except Exception as e:
        return DownloadResult(
            video_id=video_id,
            url=url,
            success=False,
            error=str(e)[:500],
        )


def parallel_download(
    videos: List[Dict[str, str]],
    output_dir: Path,
    num_workers: int = 6,
    timeout: int = 600,
    max_retries: int = 2,
) -> List[DownloadResult]:
    """
    병렬 다운로드 (대량 수집 최적화)
    
    Args:
        videos: [{"video_id": "...", "url": "..."}] 형태의 리스트
        output_dir: 출력 디렉토리
        num_workers: 워커 수 (기본 6, 대량 수집 시 8~12 추천)
        timeout: 타임아웃 (초, 기본 600초)
        max_retries: 실패 시 재시도 횟수
    
    Returns:
        다운로드 결과 리스트
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    total = len(videos)
    completed = 0
    success = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 병렬 다운로드 시작")
    print(f"{'='*60}")
    print(f"📁 출력 경로: {output_dir}")
    print(f"📦 총 파일: {total}개")
    print(f"👷 워커 수: {num_workers}")
    print(f"⏱️ 타임아웃: {timeout}초")
    print(f"🔄 최대 재시도: {max_retries}회")
    print()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(
                download_single,
                v["video_id"],
                v["url"],
                output_dir,
                timeout,
            ): v
            for v in videos
        }
        
        for future in as_completed(future_to_video):
            video = future_to_video[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    if result.skipped:
                        skipped += 1
                        status = "⏭️ 스킵"
                    else:
                        success += 1
                        size_mb = result.size_bytes / (1024 * 1024)
                        status = f"✅ 완료 ({size_mb:.1f}MB, {result.duration_sec:.1f}s)"
                else:
                    failed += 1
                    status = f"❌ 실패: {result.error[:50]}"
                
                progress = f"[{completed}/{total}]"
                print(f"{progress} {result.video_id}: {status}")
                
            except Exception as e:
                failed += 1
                results.append(DownloadResult(
                    video_id=video["video_id"],
                    url=video["url"],
                    success=False,
                    error=str(e),
                ))
                print(f"[{completed}/{total}] {video['video_id']}: ❌ 예외: {e}")
    
    total_time = time.time() - start_time
    total_size = sum(r.size_bytes for r in results if r.success and not r.skipped)
    
    print()
    print(f"{'='*60}")
    print(f"📊 다운로드 결과 요약")
    print(f"{'='*60}")
    print(f"  총 파일: {total}개")
    print(f"  총 크기: {total_size / (1024**2):.2f} MB")
    print(f"  총 시간: {total_time:.1f}초")
    print(f"  ✅ 성공: {success}개")
    print(f"  ⏭️ 스킵: {skipped}개")
    print(f"  ❌ 실패: {failed}개")
    
    if success > 0:
        print(f"  📈 평균 속도: {total_size / total_time / (1024**2):.2f} MB/s")
    
    # DB에 저장
    saved_count = save_results_to_db(results, videos)
    if saved_count > 0:
        print(f"  💾 DB 저장: {saved_count}개")
    
    return results


def save_results_to_db(results: List[DownloadResult], videos: List[Dict[str, str]]) -> int:
    """다운로드 결과를 DB에 저장"""
    if not HAS_DB:
        return 0
    
    try:
        db_path = project_root / "data" / "pade.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        saved = 0
        video_info = {v["video_id"]: v for v in videos}
        
        for result in results:
            if not result.success:
                continue
            
            info = video_info.get(result.video_id, {})
            
            # 이미 존재하는지 확인
            existing = session.query(Video).filter_by(video_id=result.video_id).first()
            if existing:
                # local_path 업데이트
                existing.local_path = result.video_path
                existing.downloaded_at = datetime.now()
                existing.status = "downloaded"
                saved += 1
                continue
            
            video = Video(
                video_id=result.video_id,
                platform="youtube",
                url=result.url,
                title=info.get("title", "Unknown"),
                local_path=result.video_path,
                downloaded_at=datetime.now(),
                status="downloaded",
            )
            session.add(video)
            saved += 1
        
        session.commit()
        session.close()
        return saved
        
    except Exception as e:
        logger.error(f"DB 저장 실패: {e}")
        return 0


def search_youtube(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """YouTube 검색"""
    import subprocess
    import json
    import shutil
    
    print(f"🔍 YouTube 검색: '{query}' (limit={limit})")
    
    # yt-dlp 경로 찾기 (venv 내 또는 시스템)
    yt_dlp_path = shutil.which("yt-dlp")
    if not yt_dlp_path:
        venv_bin = Path(sys.executable).parent
        for name in ("yt-dlp", "yt-dlp.exe"):
            candidate = venv_bin / name
            if candidate.exists():
                yt_dlp_path = str(candidate)
                break
        if not yt_dlp_path:
            logger.error("yt-dlp not found")
            return []
    
    cmd = [
        yt_dlp_path,
        f"ytsearch{limit}:{query}",
        "--flat-playlist",
        "--dump-json",
        "--quiet",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                data = json.loads(line)
                videos.append({
                    "video_id": data.get("id"),
                    "url": data.get("url") or f"https://youtube.com/watch?v={data.get('id')}",
                    "title": data.get("title", "Unknown"),
                })
        
        print(f"   → {len(videos)}개 발견")
        return videos
        
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        return []


def load_urls_from_file(filepath: str) -> List[Dict[str, str]]:
    """파일에서 URL 로드"""
    videos = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith("#"):
                # URL에서 video_id 추출
                if "youtube.com" in url or "youtu.be" in url:
                    import re
                    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
                    video_id = match.group(1) if match else url.split("/")[-1]
                else:
                    video_id = url.split("/")[-1].split("?")[0]
                
                videos.append({"video_id": video_id, "url": url})
    
    return videos


def main():
    parser = argparse.ArgumentParser(description="P-ADE 병렬 다운로드")
    
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--query", help="YouTube 검색어")
    source_group.add_argument("--urls", help="URL 파일 경로")
    source_group.add_argument("--video-ids", nargs="+", help="비디오 ID 목록")
    
    parser.add_argument("--limit", type=int, default=50, help="검색 결과 수 (기본: 50)")
    parser.add_argument("--workers", type=int, default=6, help="워커 수 (기본: 6)")
    parser.add_argument("--output", default="data/raw", help="출력 디렉토리")
    parser.add_argument("--timeout", type=int, default=300, help="타임아웃 초 (기본: 300)")
    parser.add_argument("--dry-run", action="store_true", help="실제 다운로드 없이 테스트")
    
    args = parser.parse_args()
    
    # 비디오 목록 수집
    videos = []
    
    if args.query:
        videos = search_youtube(args.query, args.limit)
    elif args.urls:
        videos = load_urls_from_file(args.urls)
    elif args.video_ids:
        videos = [
            {"video_id": vid, "url": f"https://youtube.com/watch?v={vid}"}
            for vid in args.video_ids
        ]
    
    if not videos:
        print("❌ 다운로드할 비디오가 없습니다.")
        return
    
    print(f"\n📋 다운로드 대상: {len(videos)}개")
    for i, v in enumerate(videos[:5], 1):
        title = v.get("title", v["video_id"])
        print(f"   {i}. {title[:50]}")
    if len(videos) > 5:
        print(f"   ... 외 {len(videos) - 5}개")
    
    if args.dry_run:
        print("\n🔧 Dry-run 모드: 실제 다운로드 없음")
        return
    
    # 병렬 다운로드 실행
    output_dir = Path(args.output)
    results = parallel_download(
        videos=videos,
        output_dir=output_dir,
        num_workers=args.workers,
        timeout=args.timeout,
    )
    
    # 결과 저장
    success_results = [r for r in results if r.success and not r.skipped]
    if success_results:
        print(f"\n✅ 완료! {len(success_results)}개 새 파일 다운로드됨")


if __name__ == "__main__":
    main()
