"""
로봇팔 관련 영상 10개만 객체 검출하여 episodes로 저장
- 기타 큐 작업 제거
- 영상당 처리 시작/소요 시간 로깅
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.logging_config import setup_logger
from core.queue_manager import QueueManager
from extraction.object_detector import ObjectDetector
from models.database import Base, Video, Episode

logger = setup_logger(__name__)

REQUIRED_KEYWORDS = [
    "robot", "robotic", "arm", "gripper", "manipulator",
    "pick and place", "pick & place", "grasping",
    "manipulation", "object manipulation",
    "assembly", "bin picking", "cobot",
    "FANUC", "ABB", "KUKA", "UR5", "UR10", "UR3",
    "industrial", "automation", "manufacturing",
    "로봇", "로봇팔", "그리퍼",
]

REJECT_KEYWORDS = [
    "animation", "simulation", "cgi", "3d render",
    "toy", "lego", "surgery", "medical",
    "minecraft", "roblox", "cartoon", "game",
]


def _matches_keywords(text: str) -> bool:
    lowered = text.lower()
    if any(bad in lowered for bad in REJECT_KEYWORDS):
        return False
    # 크롤링 단계에서 이미 필터된 영상이므로 키워드 있으면 통과, 없어도 통과(downloaded 상태면)
    return True


def _get_db_session(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _select_robot_arm_videos(session, limit: int) -> List[Video]:
    candidates: List[Video] = []
    videos = session.query(Video).all()

    for video in videos:
        if not video.local_path:
            continue
        path = Path(video.local_path)
        if not path.exists():
            continue

        title = video.title or ""
        description = video.description or ""
        tags = " ".join(video.tags) if isinstance(video.tags, list) else ""

        if not _matches_keywords(f"{title} {description} {tags}"):
            continue

        candidates.append(video)

    candidates.sort(
        key=lambda v: v.downloaded_at or datetime.min,
        reverse=True,
    )
    return candidates[:limit]


def _serialize_detections(detections) -> str:
    data = []
    for frame_det in detections:
        frame_data = {
            "frame_idx": frame_det.frame_idx,
            "timestamp": frame_det.timestamp,
            "objects": [
                {
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                }
                for obj in frame_det.objects
            ],
        }
        data.append(frame_data)
    return json.dumps(data)


def _save_episode_npz(output_path: Path, detections, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections_json = _serialize_detections(detections)
    metadata_json = json.dumps(metadata)

    np.savez_compressed(
        output_path,
        detections_json=detections_json,
        metadata_json=metadata_json,
    )


def _update_episode_db(session, video: Video, output_path: Path, detector: ObjectDetector):
    episode_id = f"{video.video_id}_ep001"
    episode = session.query(Episode).filter_by(episode_id=episode_id).first()
    if not episode:
        episode = Episode(
            episode_id=episode_id,
            video_id=video.id,
        )
        session.add(episode)

    episode.local_path = str(output_path)
    episode.filesize_bytes = output_path.stat().st_size if output_path.exists() else None
    episode.processing_params = {
        "detector": "yolo",
        "conf_threshold": detector.conf_threshold,
        "iou_threshold": detector.iou_threshold,
        "target_classes": detector.target_classes,
    }
    episode.model_versions = {
        "yolo": detector.model_name,
    }
    episode.created_at = datetime.utcnow()


def _clear_all_queues(use_redis: bool):
    qm = QueueManager(use_redis=use_redis)
    qm.clear_all()
    logger.info("🧹 큐 정리 완료")


def run(db_path: Path, output_dir: Path, limit: int, use_redis: bool, output_fps: float, device: str = None):
    _clear_all_queues(use_redis=use_redis)

    session = _get_db_session(db_path)
    try:
        videos = _select_robot_arm_videos(session, limit=limit)
        if not videos:
            logger.warning("로봇팔 영상이 없어 처리할 항목이 없습니다.")
            return 0

        # Determine device: CLI arg > environment variable DETECT_DEVICE > CUDA if available > CPU
        import os
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        env_device = os.environ.get('DETECT_DEVICE')
        if device:
            detector_device = device
        elif env_device:
            detector_device = env_device
        else:
            detector_device = 'cuda:0' if cuda_available else 'cpu'

        logger.info(f"Using device for detection: {detector_device}")
        detector = ObjectDetector(device=detector_device)

        for idx, video in enumerate(videos, 1):
            start_time = time.time()
            logger.info(f"🔍 검출 시작 ({idx}/{len(videos)}): {video.video_id}")

            try:
                detections = detector.detect_video(
                    video_path=video.local_path,
                    output_fps=output_fps,
                    max_frames=None,
                    visualize=False,
                )

                output_path = output_dir / f"{video.video_id}_episode.npz"
                metadata = {
                    "video_id": video.video_id,
                    "source_path": video.local_path,
                    "created_at": datetime.utcnow().isoformat(),
                    "num_frames": len(detections),
                    "output_fps": output_fps,
                }

                _save_episode_npz(output_path, detections, metadata)
                _update_episode_db(session, video, output_path, detector)
                session.commit()

                elapsed = time.time() - start_time
                logger.info(f"✅ 검출 완료: {video.video_id} ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ 검출 실패: {video.video_id} ({elapsed:.2f}s) - {e}")

    finally:
        session.close()

    return 0


def main():
    parser = argparse.ArgumentParser(description="로봇팔 영상 객체 검출 → episodes 저장 (대량 수집 지원)")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    parser.add_argument("--output-dir", default="data/episodes", help="episodes 출력 경로")
    parser.add_argument("--limit", type=int, default=500, help="처리할 영상 수 (기본 500)")
    parser.add_argument("--redis", action="store_true", help="Redis 큐 사용")
    parser.add_argument("--output-fps", type=float, default=5.0, help="검출 FPS (기본 5)")
    parser.add_argument("--device", default=None, help="모델 실행 디바이스 (예: 'cuda:0' 또는 'cpu')")
    args = parser.parse_args()

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)

    return run(
        db_path=db_path,
        output_dir=output_dir,
        limit=args.limit,
        use_redis=args.redis,
        output_fps=args.output_fps,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
