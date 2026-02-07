#!/usr/bin/env python3
"""
포즈 추출 스크립트
다운로드된 영상에서 포즈를 추출합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import argparse
from loguru import logger

from extraction.pose_estimator import MediaPipePoseEstimator
from extraction.pose_serializer import PoseSerializer


def extract_poses(video_path: Path, output_dir: Path, fps: float = 30.0, max_frames: int = None):
    """
    영상에서 포즈 추출
    
    Args:
        video_path: 입력 영상 경로
        output_dir: 출력 디렉토리
        fps: 출력 FPS
        max_frames: 최대 프레임 수 (테스트용)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_id = video_path.stem
    output_file = output_dir / f"{video_id}_pose.npz"
    
    logger.info(f"🎬 영상: {video_path}")
    logger.info(f"📁 출력: {output_file}")
    
    # 이미 처리됨?
    if output_file.exists():
        logger.info(f"⏭️ 이미 존재: {output_file}")
        return output_file
    
    # 포즈 추출기 생성
    logger.info("🚀 포즈 추출기 초기화...")
    estimator = MediaPipePoseEstimator(
        model_complexity=1,  # 1=Full (0=Lite, 2=Heavy)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_hands=True,
    )
    
    # 진행률 콜백
    def progress_callback(current, total):
        pct = current / total * 100 if total > 0 else 0
        logger.info(f"  진행: {current}/{total} ({pct:.1f}%)")
    
    # 포즈 추출
    logger.info(f"🏃 포즈 추출 시작 (FPS={fps}, max_frames={max_frames})...")
    sequence = estimator.process_video(
        str(video_path),
        output_fps=fps,
        max_frames=max_frames,
        progress_callback=progress_callback,
    )
    
    if not sequence or not sequence.frames:
        logger.error(f"❌ 포즈 추출 실패: 프레임 없음")
        return None
    
    logger.info(f"✅ {len(sequence.frames)} 프레임 추출 완료")
    
    # NumPy 형식으로 저장
    logger.info(f"💾 저장 중...")
    serializer = PoseSerializer()
    serializer.save_numpy(sequence, output_file)
    
    logger.info(f"✅ 저장 완료: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="영상에서 포즈 추출")
    parser.add_argument("video", nargs="?", help="영상 파일 경로")
    parser.add_argument("-o", "--output", default=None, help="출력 디렉토리")
    parser.add_argument("--fps", type=float, default=30.0, help="출력 FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="최대 프레임 수")
    parser.add_argument("--all", action="store_true", help="data/raw의 모든 영상 처리")
    
    args = parser.parse_args()
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent
    
    # 기본 출력 디렉토리
    output_dir = Path(args.output) if args.output else project_root / "data" / "poses"
    
    print()
    print("=" * 60)
    print("🏃 P-ADE 포즈 추출기")
    print("=" * 60)
    
    if args.all:
        # 모든 영상 처리
        raw_dir = project_root / "data" / "raw"
        videos = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.webm"))
        
        if not videos:
            print(f"❌ 영상 없음: {raw_dir}")
            return
        
        print(f"📹 영상 {len(videos)}개 발견")
        print()
        
        for i, video_path in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video_path.name}")
            extract_poses(video_path, output_dir, args.fps, args.max_frames)
            print()
    else:
        if not args.video:
            # 기본: data/raw의 첫 번째 영상
            raw_dir = project_root / "data" / "raw"
            videos = list(raw_dir.glob("*.mp4"))
            if videos:
                args.video = str(videos[0])
            else:
                print("❌ 사용법: python extract_poses.py <video_path>")
                print("   또는: python extract_poses.py --all")
                return
        
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ 파일 없음: {video_path}")
            return
        
        extract_poses(video_path, output_dir, args.fps, args.max_frames)
    
    print()
    print("=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
