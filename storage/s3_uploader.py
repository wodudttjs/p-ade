"""
S3 업로드 CLI

upload_to_s3.py 래퍼로 --input 옵션 지원
"""

import sys
import argparse
from pathlib import Path

import upload_to_s3


def main():
    parser = argparse.ArgumentParser(description="S3 업로드")
    parser.add_argument("--input", required=True, help="업로드할 파일/디렉토리")
    parser.add_argument("--bucket", type=str, help="S3 버킷 이름")
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드 없이 미리보기")
    parser.add_argument("--no-db-update", action="store_true", help="DB 업데이트 스킵")
    parser.add_argument("--prefix", type=str, help="S3 키 접두어 (기본: 입력 폴더명)")
    parser.add_argument("--from-db", action="store_true", help="DB에 등록된 episodes만 업로드")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 경로 없음: {args.input}")
        return 1

    if args.from_db:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.database import Base, Episode

        db_path = Path("data/pade.db")
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            files = []
            for ep in session.query(Episode).all():
                if ep.local_path:
                    file_path = Path(ep.local_path)
                    if file_path.exists():
                        files.append(file_path)

            if not files:
                print("📁 업로드할 episodes 파일이 없습니다.")
                return 0

            provider = upload_to_s3.get_s3_provider()
            bucket = args.bucket or upload_to_s3.get_bucket_name()
            provider.ensure_bucket(bucket)

            results = []
            for file_path in files:
                result = upload_to_s3.upload_file(
                    provider,
                    file_path,
                    bucket,
                    args.dry_run,
                    prefix=args.prefix or "episodes",
                    data_type="episodes",
                )
                results.append(result)

            if not args.no_db_update:
                upload_to_s3.update_database(results)
        finally:
            session.close()
    else:
        argv = ["upload_to_s3.py", "--input", args.input]
        if args.bucket:
            argv += ["--bucket", args.bucket]
        if args.dry_run:
            argv.append("--dry-run")
        if args.no_db_update:
            argv.append("--no-db-update")
        if args.prefix:
            argv += ["--prefix", args.prefix]

        sys.argv = argv
        upload_to_s3.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
