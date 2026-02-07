"""
로봇팔 관련 영상만 남기고 DB/파일 정리
"""

import argparse
from pathlib import Path
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Video, Episode, ProcessingJob, CloudFile

REQUIRED_KEYWORDS = [
    "robot arm", "robotic arm", "robot gripper",
    "pick and place", "pick & place", "grasping",
    "manipulation", "object manipulation",
    "assembly", "bin picking",
]

REJECT_KEYWORDS = [
    "animation", "simulation", "cgi", "3d render",
    "toy", "lego", "surgery", "medical",
]

MIN_DURATION_SEC = 10
MAX_DURATION_SEC = 300


def _matches_keywords(text: str) -> bool:
    lowered = text.lower()
    if any(bad in lowered for bad in REJECT_KEYWORDS):
        return False
    return any(key in lowered for key in REQUIRED_KEYWORDS)


def _keep_video(video: Video) -> bool:
    duration = video.duration_sec or 0
    if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
        return False

    title = video.title or ""
    description = video.description or ""
    tags = " ".join(video.tags) if isinstance(video.tags, list) else ""
    return _matches_keywords(f"{title} {description} {tags}")


def cleanup(db_path: Path, episodes_dir: Path, apply: bool = False):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        videos: List[Video] = session.query(Video).all()
        remove_video_ids = [v.id for v in videos if not _keep_video(v)]

        # 삭제 대상 에피소드 수집
        episodes = session.query(Episode).filter(Episode.video_id.in_(remove_video_ids)).all()
        episode_files = [Path(e.local_path) for e in episodes if e.local_path]

        if apply:
            # CloudFile 정리
            session.query(CloudFile).filter(CloudFile.video_id.in_(remove_video_ids)).delete(synchronize_session=False)
            session.query(CloudFile).filter(CloudFile.episode_id.in_([e.id for e in episodes])).delete(synchronize_session=False)

            # ProcessingJob 정리
            session.query(ProcessingJob).filter(ProcessingJob.video_id.in_([v.video_id for v in videos if v.id in remove_video_ids])).delete(synchronize_session=False)

            # Episode/Video 정리
            session.query(Episode).filter(Episode.video_id.in_(remove_video_ids)).delete(synchronize_session=False)
            session.query(Video).filter(Video.id.in_(remove_video_ids)).delete(synchronize_session=False)
            session.commit()

            # 파일 삭제
            for f in episode_files:
                if f.exists():
                    f.unlink()
        else:
            print(f"삭제 예정 비디오: {len(remove_video_ids)}")
            print(f"삭제 예정 에피소드: {len(episodes)}")

        # DB에 없는 에피소드 파일 정리
        if apply and episodes_dir.exists():
            keep_paths = {Path(e.local_path).resolve() for e in session.query(Episode).all() if e.local_path}
            for f in episodes_dir.glob("*.npz"):
                if f.resolve() not in keep_paths:
                    f.unlink()
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="로봇팔 관련 데이터 정리")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    parser.add_argument("--episodes-dir", default="data/episodes", help="episodes 디렉토리")
    parser.add_argument("--apply", action="store_true", help="실제 삭제 수행")
    args = parser.parse_args()

    db_path = Path(args.db)
    episodes_dir = Path(args.episodes_dir)
    cleanup(db_path, episodes_dir, apply=args.apply)


if __name__ == "__main__":
    main()
