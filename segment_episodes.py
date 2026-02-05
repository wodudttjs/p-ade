#!/usr/bin/env python3
"""
에피소드 분할 스크립트

포즈 데이터를 분석하여 동작 구간(에피소드)으로 분할합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Video, Episode


@dataclass
class EpisodeSegment:
    """분할된 에피소드 구간"""
    start_frame: int
    end_frame: int
    duration_frames: int
    confidence_score: float
    quality_score: float
    jittering_score: float


def calculate_motion_energy(poses: np.ndarray) -> np.ndarray:
    """
    프레임 간 움직임 에너지 계산
    
    Args:
        poses: [T, 33, 3] 포즈 배열
    
    Returns:
        [T-1] 움직임 에너지 배열
    """
    # 연속 프레임 간 차이
    diff = np.diff(poses, axis=0)
    
    # L2 거리로 움직임 크기 계산
    motion = np.linalg.norm(diff, axis=2)  # [T-1, 33]
    
    # 전체 관절의 평균 움직임
    energy = motion.mean(axis=1)  # [T-1]
    
    return energy


def calculate_pose_confidence(confidence: np.ndarray, window: int = 30) -> np.ndarray:
    """
    슬라이딩 윈도우로 평균 신뢰도 계산
    """
    if len(confidence) < window:
        return np.full(len(confidence), confidence.mean())
    
    # 이동 평균
    kernel = np.ones(window) / window
    smoothed = np.convolve(confidence, kernel, mode='same')
    
    return smoothed


def calculate_jittering(poses: np.ndarray, window: int = 5) -> np.ndarray:
    """
    지터링(불안정한 움직임) 점수 계산
    
    높은 주파수의 움직임 변화를 감지
    """
    if len(poses) < window * 2:
        return np.zeros(len(poses))
    
    # 가속도 (움직임의 변화율)
    velocity = np.diff(poses, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    # 가속도의 크기
    acc_magnitude = np.linalg.norm(acceleration, axis=2).mean(axis=1)
    
    # 패딩
    jitter = np.zeros(len(poses))
    jitter[1:-1] = acc_magnitude
    
    return jitter


def segment_by_motion(
    motion_energy: np.ndarray,
    min_frames: int = 30,
    max_frames: int = 300,
    motion_threshold: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    움직임 에너지 기반 에피소드 분할
    
    Args:
        motion_energy: 움직임 에너지 배열
        min_frames: 최소 에피소드 길이
        max_frames: 최대 에피소드 길이
        motion_threshold: 움직임 임계값
    
    Returns:
        [(start, end), ...] 에피소드 구간 리스트
    """
    # 움직임이 있는 구간 찾기
    is_moving = motion_energy > motion_threshold
    
    segments = []
    start = None
    
    for i, moving in enumerate(is_moving):
        if moving and start is None:
            start = i
        elif not moving and start is not None:
            if i - start >= min_frames:
                # 최대 길이로 분할
                seg_start = start
                while seg_start < i:
                    seg_end = min(seg_start + max_frames, i)
                    if seg_end - seg_start >= min_frames:
                        segments.append((seg_start, seg_end))
                    seg_start = seg_end
            start = None
    
    # 마지막 구간 처리
    if start is not None and len(is_moving) - start >= min_frames:
        seg_start = start
        while seg_start < len(is_moving):
            seg_end = min(seg_start + max_frames, len(is_moving))
            if seg_end - seg_start >= min_frames:
                segments.append((seg_start, seg_end))
            seg_start = seg_end
    
    return segments


def segment_fixed_length(
    total_frames: int,
    segment_length: int = 150,
    overlap: int = 30,
) -> List[Tuple[int, int]]:
    """
    고정 길이로 분할 (오버랩 허용)
    """
    segments = []
    start = 0
    step = segment_length - overlap
    
    while start + segment_length <= total_frames:
        segments.append((start, start + segment_length))
        start += step
    
    # 마지막 구간
    if start < total_frames and total_frames - start >= segment_length // 2:
        segments.append((start, total_frames))
    
    return segments


def segment_poses(
    pose_file: Path,
    min_frames: int = 30,
    max_frames: int = 300,
    min_confidence: float = 0.5,
    method: str = "motion",
) -> List[EpisodeSegment]:
    """
    포즈 파일에서 에피소드 분할
    
    Args:
        pose_file: 포즈 npz 파일 경로
        min_frames: 최소 에피소드 길이
        max_frames: 최대 에피소드 길이
        min_confidence: 최소 신뢰도
        method: 분할 방식 ("motion" 또는 "fixed")
    
    Returns:
        에피소드 리스트
    """
    data = np.load(pose_file, allow_pickle=True)
    
    body = data['body']  # [T, 33, 3]
    confidence = data['confidence']  # [T]
    
    T = len(body)
    logger.info(f"포즈 데이터: {T} 프레임")
    
    # 움직임 에너지 계산
    motion_energy = calculate_motion_energy(body)
    
    # 지터링 계산
    jittering = calculate_jittering(body)
    
    # 분할
    if method == "motion":
        segments = segment_by_motion(motion_energy, min_frames, max_frames)
    else:
        segments = segment_fixed_length(T, max_frames, min_frames)
    
    logger.info(f"분할된 구간: {len(segments)}개")
    
    # 에피소드 생성
    episodes = []
    for start, end in segments:
        # 구간 신뢰도
        seg_confidence = confidence[start:end].mean()
        
        if seg_confidence < min_confidence:
            logger.debug(f"구간 {start}-{end}: 신뢰도 부족 ({seg_confidence:.3f})")
            continue
        
        # 품질 점수 계산
        seg_jitter = jittering[start:end].mean()
        quality_score = seg_confidence * (1 - min(seg_jitter * 10, 1))
        
        episode = EpisodeSegment(
            start_frame=start,
            end_frame=end,
            duration_frames=end - start,
            confidence_score=seg_confidence,
            quality_score=quality_score,
            jittering_score=seg_jitter,
        )
        episodes.append(episode)
    
    return episodes


def save_episodes_to_db(
    video_id: str,
    episodes: List[EpisodeSegment],
    db_path: str,
):
    """에피소드를 DB에 저장"""
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 비디오 찾기
    video = session.query(Video).filter(Video.video_id == video_id).first()
    
    if not video:
        logger.warning(f"비디오 없음: {video_id}")
        session.close()
        return 0
    
    saved = 0
    for i, ep in enumerate(episodes):
        episode_id = f"{video_id}_ep{i:03d}"
        
        # 중복 체크
        existing = session.query(Episode).filter(Episode.episode_id == episode_id).first()
        if existing:
            logger.debug(f"이미 존재: {episode_id}")
            continue
        
        episode = Episode(
            video_id=video.id,
            episode_id=episode_id,
            start_frame=ep.start_frame,
            end_frame=ep.end_frame,
            duration_frames=ep.duration_frames,
            confidence_score=ep.confidence_score,
            quality_score=ep.quality_score,
            jittering_score=ep.jittering_score,
            job_key=episode_id,
        )
        session.add(episode)
        saved += 1
    
    session.commit()
    session.close()
    
    logger.info(f"✅ {saved}개 에피소드 저장: {video_id}")
    return saved


def process_pose_file(
    pose_file: Path,
    db_path: str,
    min_frames: int = 30,
    max_frames: int = 300,
    min_confidence: float = 0.5,
    method: str = "motion",
):
    """단일 포즈 파일 처리"""
    video_id = pose_file.stem.replace("_pose", "")
    
    logger.info(f"🎬 처리: {pose_file.name}")
    
    # 에피소드 분할
    episodes = segment_poses(
        pose_file,
        min_frames=min_frames,
        max_frames=max_frames,
        min_confidence=min_confidence,
        method=method,
    )
    
    if not episodes:
        logger.warning(f"에피소드 없음: {video_id}")
        return
    
    logger.info(f"📊 {len(episodes)}개 에피소드 생성")
    
    # 품질 통계
    qualities = [ep.quality_score for ep in episodes]
    logger.info(f"  품질: 평균 {np.mean(qualities):.3f}, 최소 {min(qualities):.3f}, 최대 {max(qualities):.3f}")
    
    # DB 저장
    save_episodes_to_db(video_id, episodes, db_path)


def main():
    parser = argparse.ArgumentParser(description="포즈 데이터 에피소드 분할")
    parser.add_argument("pose", nargs="?", help="포즈 파일 경로")
    parser.add_argument("--all", action="store_true", help="data/poses의 모든 파일 처리")
    parser.add_argument("--db", default=None, help="데이터베이스 경로")
    parser.add_argument("--min-frames", type=int, default=30, help="최소 에피소드 길이")
    parser.add_argument("--max-frames", type=int, default=300, help="최대 에피소드 길이")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="최소 신뢰도")
    parser.add_argument("--method", choices=["motion", "fixed"], default="motion", help="분할 방식")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    db_path = args.db or str(project_root / "data" / "pade.db")
    
    print()
    print("=" * 60)
    print("✂️ P-ADE 에피소드 분할기")
    print("=" * 60)
    
    if args.all:
        poses_dir = project_root / "data" / "poses"
        pose_files = list(poses_dir.glob("*_pose.npz"))
        
        if not pose_files:
            print(f"❌ 포즈 파일 없음: {poses_dir}")
            return
        
        print(f"📁 포즈 파일 {len(pose_files)}개 발견")
        print()
        
        total_episodes = 0
        for i, pose_file in enumerate(pose_files, 1):
            print(f"[{i}/{len(pose_files)}] {pose_file.name}")
            process_pose_file(
                pose_file, db_path,
                args.min_frames, args.max_frames,
                args.min_confidence, args.method
            )
            print()
    else:
        if not args.pose:
            poses_dir = project_root / "data" / "poses"
            pose_files = list(poses_dir.glob("*_pose.npz"))
            if pose_files:
                args.pose = str(pose_files[0])
            else:
                print("❌ 사용법: python segment_episodes.py <pose_file>")
                print("   또는: python segment_episodes.py --all")
                return
        
        pose_file = Path(args.pose)
        if not pose_file.exists():
            print(f"❌ 파일 없음: {pose_file}")
            return
        
        process_pose_file(
            pose_file, db_path,
            args.min_frames, args.max_frames,
            args.min_confidence, args.method
        )
    
    print()
    print("=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
