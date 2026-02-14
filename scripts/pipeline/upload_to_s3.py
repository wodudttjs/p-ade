#!/usr/bin/env python
"""
P-ADE S3 업로드 스크립트

포즈 데이터(.npz)와 에피소드를 S3에 업로드합니다.

Usage:
    python upload_to_s3.py --all                      # 모든 포즈 파일 업로드
    python upload_to_s3.py --file data/poses/xxx.npz  # 특정 파일 업로드
    python upload_to_s3.py --input data/episodes/     # 특정 디렉토리 업로드
    python upload_to_s3.py --dry-run --all            # 업로드할 파일 미리보기
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from config.settings import Config

logger = setup_logger(__name__)


def get_bucket_name() -> str:
    """S3 버킷 이름 가져오기"""
    return os.getenv("S3_BUCKET", Config.AWS_S3_BUCKET)


def get_s3_provider():
    """S3 Provider 인스턴스 생성"""
    from storage.providers.s3_provider import S3Provider
    
    return S3Provider(
        region=os.getenv("AWS_REGION", "ap-northeast-2"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),  # LocalStack 등
    )


def find_pose_files() -> List[Path]:
    """업로드할 포즈 파일 찾기"""
    poses_dir = project_root / "data" / "poses"
    if not poses_dir.exists():
        return []
    return list(poses_dir.glob("*.npz"))


def generate_s3_key(local_path: Path, prefix: str = "poses") -> str:
    """S3 키 생성
    
    형식: poses/YYYY/MM/DD/{video_id}_pose.npz
    """
    today = datetime.now()
    date_prefix = f"{today.year}/{today.month:02d}/{today.day:02d}"
    return f"{prefix}/{date_prefix}/{local_path.name}"


def get_file_metadata(local_path: Path, data_type: str = "pose") -> Dict[str, str]:
    """파일 메타데이터 생성"""
    stat = local_path.stat()
    return {
        "original_filename": local_path.name,
        "upload_timestamp": datetime.now().isoformat(),
        "file_size": str(stat.st_size),
        "project": "p-ade",
        "data_type": data_type,
    }


def upload_file(
    provider,
    local_path: Path,
    bucket: str,
    dry_run: bool = False,
    prefix: str = "poses",
    data_type: str = "pose",
) -> Dict:
    """단일 파일 업로드"""
    # 파일 존재 확인
    if not local_path.exists():
        logger.error(f"❌ 파일 없음: {local_path}")
        return {
            "local_path": str(local_path),
            "status": "error",
            "error": f"File not found: {local_path}",
        }
    
    s3_key = generate_s3_key(local_path, prefix=prefix)
    metadata = get_file_metadata(local_path, data_type=data_type)
    file_size = local_path.stat().st_size
    
    result = {
        "local_path": str(local_path),
        "s3_key": s3_key,
        "bucket": bucket,
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
    }
    
    if dry_run:
        result["status"] = "dry_run"
        result["uri"] = f"s3://{bucket}/{s3_key}"
        logger.info(f"[DRY-RUN] 업로드 예정: {local_path.name} -> s3://{bucket}/{s3_key}")
        return result
    
    try:
        upload_result = provider.upload_file(
            local_path=str(local_path),
            remote_key=s3_key,
            bucket=bucket,
            metadata=metadata,
            storage_class="STANDARD",
        )
        
        result["status"] = upload_result.status.value
        result["uri"] = upload_result.uri
        result["etag"] = upload_result.etag
        result["sha256"] = upload_result.sha256
        
        if upload_result.status.value == "completed":
            logger.info(f"✅ 업로드 완료: {local_path.name} -> {upload_result.uri}")
        elif upload_result.status.value == "skipped":
            logger.info(f"⏭️ 이미 존재 (skip): {local_path.name}")
        else:
            logger.error(f"❌ 업로드 실패: {local_path.name} - {upload_result.error_message}")
            result["error"] = upload_result.error_message
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"❌ 업로드 오류: {local_path.name} - {e}")
        
    return result


def _parse_episode_ids(file_path: Path) -> Dict[str, str]:
    stem = file_path.stem
    base = stem[:-5] if stem.endswith("_pose") else stem
    if "_ep" in base:
        video_id = base.split("_ep")[0]
        episode_id = base
    else:
        video_id = base
        episode_id = f"{base}_ep001"
    return {"video_id": video_id, "episode_id": episode_id}


def register_episodes_in_db(files: List[Path]):
    """episodes 파일을 DB에 등록 (local_path, filesize 업데이트)"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.database import Base, Video, Episode
    except Exception as e:
        logger.warning(f"DB 모듈 로드 실패, 등록 스킵: {e}")
        return

    db_path = project_root / "data" / "pade.db"
    if not db_path.exists():
        logger.warning("DB 파일 없음, 등록 스킵")
        return

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for file_path in files:
            ids = _parse_episode_ids(file_path)
            video = session.query(Video).filter_by(video_id=ids["video_id"]).first()
            if not video:
                video = Video(
                    video_id=ids["video_id"],
                    platform="youtube",
                    url="",
                    status="processed",
                )
                session.add(video)
                session.flush()

            episode = session.query(Episode).filter_by(episode_id=ids["episode_id"]).first()
            if not episode:
                episode = Episode(
                    episode_id=ids["episode_id"],
                    video_id=video.id,
                )
                session.add(episode)

            episode.local_path = str(file_path)
            if file_path.exists():
                episode.filesize_bytes = file_path.stat().st_size

        session.commit()
    finally:
        session.close()


def update_database(results: List[Dict]):
    """업로드 결과를 DB에 반영"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.database import Base, Video, Episode
    except Exception as e:
        logger.warning(f"DB 모듈 로드 실패, 업데이트 스킵: {e}")
        return

    db_path = project_root / "data" / "pade.db"
    if not db_path.exists():
        logger.warning("DB 파일 없음, 업데이트 스킵")
        return

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    updated = 0
    try:
        for result in results:
            if result.get("status") in ["completed", "skipped"]:
                file_path = Path(result["local_path"])
                ids = _parse_episode_ids(file_path)

                video = session.query(Video).filter_by(video_id=ids["video_id"]).first()
                if not video:
                    video = Video(
                        video_id=ids["video_id"],
                        platform="youtube",
                        url="",
                        status="processed",
                    )
                    session.add(video)
                    session.flush()

                episode = session.query(Episode).filter_by(episode_id=ids["episode_id"]).first()
                if not episode:
                    episode = Episode(
                        episode_id=ids["episode_id"],
                        video_id=video.id,
                    )
                    session.add(episode)

                episode.cloud_path = result.get("uri")
                episode.uploaded_at = datetime.now()
                updated += 1

        session.commit()
    finally:
        session.close()

    logger.info(f"📝 DB 업데이트: {updated}개 에피소드")


def main():
    parser = argparse.ArgumentParser(description="P-ADE S3 업로드")
    parser.add_argument("--all", action="store_true", help="모든 포즈 파일 업로드")
    parser.add_argument("--file", type=str, help="특정 파일 업로드")
    parser.add_argument("--input", type=str, help="파일 또는 디렉토리 업로드")
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드 없이 미리보기")
    parser.add_argument("--bucket", type=str, help="S3 버킷 이름 (기본: 환경변수)")
    parser.add_argument("--no-db-update", action="store_true", help="DB 업데이트 스킵")
    parser.add_argument("--prefix", type=str, help="S3 키 접두어 (기본: 입력 폴더명)")
    
    args = parser.parse_args()
    
    if not args.all and not args.file and not args.input:
        parser.print_help()
        print("\n❌ --all, --file 또는 --input 옵션을 지정해주세요.")
        sys.exit(1)
        
    # 파일 목록 수집
    prefix = "poses"
    data_type = "pose"

    if args.all:
        files = find_pose_files()
        if not files:
            print("📁 업로드할 포즈 파일이 없습니다.")
            sys.exit(0)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 경로 없음: {args.input}")
            sys.exit(1)
        if input_path.is_file():
            files = [input_path]
        else:
            files = [p for p in input_path.rglob("*") if p.is_file()]
        if not files:
            print(f"📁 업로드할 파일이 없습니다: {args.input}")
            sys.exit(0)
        if args.prefix:
            prefix = args.prefix
        else:
            prefix = input_path.name or "input"
        data_type = prefix
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 파일 없음: {args.file}")
            sys.exit(1)
        files = [file_path]
        if args.prefix:
            prefix = args.prefix
        else:
            prefix = file_path.parent.name if file_path.parent.name else "poses"
        data_type = prefix
        
    # 버킷 이름
    bucket = args.bucket or get_bucket_name()
    
    print("=" * 60)
    print("🚀 P-ADE S3 업로드")
    print("=" * 60)
    print(f"📁 업로드 파일: {len(files)}개")
    print(f"🪣 버킷: {bucket}")
    print(f"🔧 Dry-run: {args.dry_run}")
    print()
    
    # S3 Provider 초기화
    if not args.dry_run:
        try:
            provider = get_s3_provider()
            # 버킷 확인/생성
            provider.ensure_bucket(bucket)
            logger.info(f"🪣 버킷 준비 완료: {bucket}")
        except ImportError:
            print("❌ boto3가 설치되지 않았습니다.")
            print("   pip install boto3")
            sys.exit(1)
        except Exception as e:
            print(f"❌ S3 연결 실패: {e}")
            print("   AWS 자격 증명을 확인하세요:")
            print("   - AWS_ACCESS_KEY_ID")
            print("   - AWS_SECRET_ACCESS_KEY")
            print("   - AWS_REGION")
            sys.exit(1)
    else:
        provider = None
        
    # 업로드 실행
    results = []
    total_size = 0
    
    if not args.no_db_update and data_type in ["episodes", "episode"]:
        register_episodes_in_db(files)

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path.name}")
        result = upload_file(
            provider,
            file_path,
            bucket,
            args.dry_run,
            prefix=prefix,
            data_type=data_type,
        )
        results.append(result)
        total_size += result.get("size_bytes", 0)
        
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 업로드 결과 요약")
    print("=" * 60)
    
    completed = sum(1 for r in results if r.get("status") == "completed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") in ["error", "failed"])
    dry_run_count = sum(1 for r in results if r.get("status") == "dry_run")
    
    print(f"  총 파일: {len(results)}개")
    print(f"  총 크기: {total_size / (1024*1024):.2f} MB")
    
    if args.dry_run:
        print(f"  미리보기: {dry_run_count}개")
    else:
        print(f"  ✅ 완료: {completed}개")
        print(f"  ⏭️ 스킵: {skipped}개")
        print(f"  ❌ 실패: {failed}개")
        
        # DB 업데이트
        if not args.no_db_update and (completed > 0 or skipped > 0):
            update_database(results)
    
    print()
    print("✅ 완료!")
    

if __name__ == "__main__":
    main()
