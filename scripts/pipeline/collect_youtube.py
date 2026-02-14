#!/usr/bin/env python
"""
YouTube 영상 수집 스크립트

키워드로 YouTube 영상을 검색하고 다운로드하는 간단한 CLI
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
from typing import List, Optional

import yt_dlp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Video, ProcessingJob
from core.logging_config import setup_logger

logger = setup_logger(__name__)


def search_youtube(keyword: str, max_results: int = 10) -> List[dict]:
    """
    YouTube에서 키워드로 검색
    
    Args:
        keyword: 검색 키워드
        max_results: 최대 결과 수
    
    Returns:
        비디오 정보 리스트
    """
    logger.info(f"🔍 YouTube 검색: '{keyword}' (최대 {max_results}개)")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }
    
    search_url = f"ytsearch{max_results}:{keyword}"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(search_url, download=False)
    
    videos = []
    if result and 'entries' in result:
        for entry in result['entries']:
            if entry:
                videos.append({
                    'video_id': entry.get('id'),
                    'title': entry.get('title'),
                    'url': entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'channel': entry.get('channel') or entry.get('uploader'),
                    'duration': entry.get('duration'),
                })
    
    logger.info(f"✅ {len(videos)}개 비디오 발견")
    return videos


def get_video_info(video_url: str) -> Optional[dict]:
    """비디오 상세 정보 가져오기"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return {
                'video_id': info.get('id'),
                'title': info.get('title'),
                'description': info.get('description'),
                'duration_sec': info.get('duration'),
                'upload_date': info.get('upload_date'),
                'channel_id': info.get('channel_id'),
                'channel_name': info.get('channel') or info.get('uploader'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'thumbnail_url': info.get('thumbnail'),
                'tags': info.get('tags', []),
            }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None


def download_video(video_url: str, output_dir: Path, quality: str = "720p") -> Optional[str]:
    """
    비디오 다운로드
    
    Args:
        video_url: 비디오 URL
        output_dir: 출력 디렉토리
        quality: 품질 (360p, 720p, 1080p)
    
    Returns:
        다운로드된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # mp4 단일 파일로 다운로드 (ffmpeg 불필요)
    quality_map = {
        "360p": "best[height<=360][ext=mp4]/best[height<=360]",
        "720p": "best[height<=720][ext=mp4]/best[height<=720]",
        "1080p": "best[height<=1080][ext=mp4]/best[height<=1080]",
    }
    
    ydl_opts = {
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'format': quality_map.get(quality, quality_map["720p"]),
        'quiet': False,
        'no_warnings': False,
        'retries': 3,
        'nocheckcertificate': True,
        'no_check_certificate': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get('id')
            ext = info.get('ext', 'mp4')
            filepath = output_dir / f"{video_id}.{ext}"
            
            if filepath.exists():
                return str(filepath)
            
            # mp4로 변환된 경우
            mp4_path = output_dir / f"{video_id}.mp4"
            if mp4_path.exists():
                return str(mp4_path)
            
            return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def save_to_database(videos: List[dict], db_path: str = "data/pade.db"):
    """DB에 비디오 정보 저장"""
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    saved_count = 0
    for video_data in videos:
        # 중복 체크
        existing = session.query(Video).filter_by(
            video_id=video_data['video_id']
        ).first()
        
        if existing:
            logger.info(f"⏭️ 이미 존재: {video_data['video_id']}")
            continue
        
        video = Video(
            video_id=video_data['video_id'],
            platform='youtube',
            url=video_data.get('url', ''),
            title=video_data.get('title'),
            description=video_data.get('description'),
            duration_sec=video_data.get('duration_sec'),
            channel_id=video_data.get('channel_id'),
            channel_name=video_data.get('channel_name'),
            view_count=video_data.get('view_count'),
            like_count=video_data.get('like_count'),
            thumbnail_url=video_data.get('thumbnail_url'),
            tags=video_data.get('tags'),
            status='discovered',
            discovered_at=datetime.utcnow(),
        )
        session.add(video)
        saved_count += 1
        logger.info(f"💾 저장: {video_data['video_id']} - {video_data.get('title', '')[:50]}")
    
    session.commit()
    session.close()
    
    logger.info(f"✅ {saved_count}개 비디오 DB 저장 완료")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="YouTube 영상 수집")
    parser.add_argument("keyword", help="검색 키워드")
    parser.add_argument("-n", "--max-results", type=int, default=50, help="최대 결과 수 (기본: 50)")
    parser.add_argument("-d", "--download", action="store_true", help="비디오 다운로드")
    parser.add_argument("-q", "--quality", default="720p", choices=["360p", "720p", "1080p"], help="다운로드 품질")
    parser.add_argument("-o", "--output", default="data/raw", help="출력 디렉토리")
    parser.add_argument("--db", default="data/pade.db", help="데이터베이스 경로")
    parser.add_argument("--min-duration", type=int, default=None, help="최소 영상 길이 (초)")
    parser.add_argument("--max-duration", type=int, default=None, help="최대 영상 길이 (초)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 P-ADE YouTube 수집기")
    print("=" * 60)
    print(f"키워드: {args.keyword}")
    print(f"최대 결과: {args.max_results}")
    print(f"다운로드: {'예' if args.download else '아니오'}")
    if args.min_duration or args.max_duration:
        min_sec = args.min_duration or 0
        max_sec = args.max_duration or float('inf')
        print(f"영상 길이: {min_sec}초 ~ {max_sec if max_sec != float('inf') else '제한없음'}초")
    print("=" * 60)
    
    # 1. 검색 (필터링을 위해 더 많이 가져오기)
    search_count = args.max_results * 3 if (args.min_duration or args.max_duration) else args.max_results
    videos = search_youtube(args.keyword, search_count)
    
    if not videos:
        print("❌ 검색 결과가 없습니다.")
        return
    
    # 2. 상세 정보 가져오기 및 필터링
    print("\n📋 비디오 상세 정보 수집 중...")
    detailed_videos = []
    for video in videos:
        if len(detailed_videos) >= args.max_results:
            break
            
        print(f"  - {video['video_id']}: {video['title'][:50]}...")
        info = get_video_info(video['url'])
        
        if info:
            duration = info.get('duration_sec', 0) or 0
            
            # 길이 필터링
            if args.min_duration and duration < args.min_duration:
                print(f"    ⏭️ 건너뜀: {duration}초 (최소 {args.min_duration}초 미만)")
                continue
            if args.max_duration and duration > args.max_duration:
                print(f"    ⏭️ 건너뜀: {duration}초 (최대 {args.max_duration}초 초과)")
                continue
            
            print(f"    ✅ 선택: {duration}초")
            detailed_videos.append(info)
    
    if not detailed_videos:
        print("❌ 조건에 맞는 영상이 없습니다.")
        return
    
    print(f"\n✅ {len(detailed_videos)}개 영상 선택됨")
    
    # 3. DB 저장
    print(f"\n💾 데이터베이스 저장 중... ({args.db})")
    save_to_database(detailed_videos, args.db)
    
    # 4. 다운로드 (옵션)
    if args.download:
        output_dir = Path(args.output)
        print(f"\n📥 비디오 다운로드 중... ({output_dir})")
        
        for video in detailed_videos:
            video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
            print(f"\n⬇️ 다운로드: {video['title'][:50]}...")
            
            filepath = download_video(video_url, output_dir, args.quality)
            
            if filepath:
                print(f"   ✅ 완료: {filepath}")
            else:
                print(f"   ❌ 실패")
    
    print("\n" + "=" * 60)
    print("✅ 수집 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
