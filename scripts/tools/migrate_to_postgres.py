"""
SQLite → PostgreSQL 마이그레이션 스크립트

사용:
    python scripts/tools/migrate_to_postgres.py \
        --sqlite data/pade.db \
        --postgres postgresql://pade:pade@localhost:5432/pade \
        --verify
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

try:
    import sqlalchemy as sa
    from sqlalchemy import text
except ImportError:
    print("sqlalchemy가 설치되지 않았습니다: pip install sqlalchemy psycopg2-binary")
    sys.exit(1)

# ─── 스키마 DDL ───────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS video_registry (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(64) UNIQUE NOT NULL,
    url_hash VARCHAR(64) NOT NULL,
    url TEXT,
    platform VARCHAR(32),
    status VARCHAR(16) DEFAULT 'collected',
    run_id VARCHAR(64),
    collected_at TIMESTAMP DEFAULT NOW(),
    quality_score FLOAT,
    rejection_reason TEXT,
    s3_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_registry_video_id ON video_registry(video_id);
CREATE INDEX IF NOT EXISTS idx_video_registry_url_hash ON video_registry(url_hash);
CREATE INDEX IF NOT EXISTS idx_video_registry_status ON video_registry(status);
CREATE INDEX IF NOT EXISTS idx_video_registry_collected_at ON video_registry(collected_at);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    target_count INT,
    crawled INT DEFAULT 0,
    downloaded INT DEFAULT 0,
    processed INT DEFAULT 0,
    passed INT DEFAULT 0,
    uploaded INT DEFAULT 0,
    status VARCHAR(16) DEFAULT 'running'
);
"""


# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def _get_tables(engine: sa.Engine) -> List[str]:
    """엔진에서 테이블 목록 반환"""
    insp = sa.inspect(engine)
    return insp.get_table_names()


def _copy_table(
    src_engine: sa.Engine,
    dst_engine: sa.Engine,
    table_name: str,
) -> int:
    """테이블 데이터 복사, 복사된 행 수 반환"""
    with src_engine.connect() as src_conn:
        rows: List[Dict[str, Any]] = [
            dict(r._mapping)
            for r in src_conn.execute(text(f"SELECT * FROM {table_name}"))
        ]

    if not rows:
        logger.info(f"  {table_name}: 데이터 없음 (0행)")
        return 0

    with dst_engine.begin() as dst_conn:
        # PostgreSQL UPSERT (id 충돌 무시)
        meta = sa.MetaData()
        meta.reflect(bind=dst_engine, only=[table_name])
        tbl = meta.tables[table_name]

        insert_stmt = sa.dialects.postgresql.insert(tbl)
        on_conflict = insert_stmt.on_conflict_do_nothing()
        dst_conn.execute(on_conflict, rows)

    logger.info(f"  {table_name}: {len(rows)}행 복사 완료")
    return len(rows)


# ─── 메인 함수 ────────────────────────────────────────────────────────────────

def create_schema(pg_engine: sa.Engine) -> None:
    """PostgreSQL 스키마 생성"""
    logger.info("스키마 생성 중...")
    with pg_engine.begin() as conn:
        for stmt in SCHEMA_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("스키마 생성 완료")


def migrate(
    sqlite_url: str,
    postgres_url: str,
    verify: bool = False,
) -> bool:
    """
    SQLite → PostgreSQL 데이터 마이그레이션

    Returns:
        True if successful
    """
    logger.info(f"SQLite: {sqlite_url}")
    logger.info(f"PostgreSQL: {postgres_url}")

    sqlite_engine = sa.create_engine(sqlite_url)
    pg_engine = sa.create_engine(postgres_url)

    # 1. 스키마 생성
    create_schema(pg_engine)

    # 2. SQLite 테이블 목록
    src_tables = _get_tables(sqlite_engine)
    dst_tables = _get_tables(pg_engine)
    logger.info(f"SQLite 테이블: {src_tables}")

    # 3. 데이터 복사 (공통 테이블만)
    total_rows = 0
    for table in src_tables:
        if table in dst_tables:
            try:
                n = _copy_table(sqlite_engine, pg_engine, table)
                total_rows += n
            except Exception as e:
                logger.error(f"  {table} 복사 실패: {e}")
        else:
            logger.warning(f"  {table}: PostgreSQL에 없는 테이블 — 건너뜀")

    logger.info(f"총 {total_rows}행 복사 완료")

    # 4. 검증
    if verify:
        logger.info("검증 중...")
        all_ok = True
        for table in src_tables:
            if table not in dst_tables:
                continue
            with sqlite_engine.connect() as c:
                src_count = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            with pg_engine.connect() as c:
                dst_count = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            status = "OK" if src_count == dst_count else "MISMATCH"
            logger.info(f"  {table}: SQLite={src_count}, PostgreSQL={dst_count} [{status}]")
            if status != "OK":
                all_ok = False
        return all_ok

    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 마이그레이션")
    parser.add_argument(
        "--sqlite",
        default="data/pade.db",
        help="SQLite DB 파일 경로 (기본: data/pade.db)",
    )
    parser.add_argument(
        "--postgres",
        default="postgresql://pade:pade@localhost:5432/pade",
        help="PostgreSQL 연결 URL",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="마이그레이션 후 행 수 검증",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="스키마만 생성 (데이터 복사 안 함)",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    sqlite_url = f"sqlite:///{sqlite_path}"

    if args.schema_only:
        pg_engine = sa.create_engine(args.postgres)
        create_schema(pg_engine)
        logger.info("스키마 생성 완료")
        return

    if not sqlite_path.exists() and "sqlite" in sqlite_url:
        logger.warning(f"SQLite 파일 없음: {sqlite_path} — 스키마만 생성합니다")
        pg_engine = sa.create_engine(args.postgres)
        create_schema(pg_engine)
        return

    success = migrate(
        sqlite_url=sqlite_url,
        postgres_url=args.postgres,
        verify=args.verify,
    )

    if not success:
        logger.error("마이그레이션 검증 실패")
        sys.exit(1)

    logger.info("마이그레이션 성공")


if __name__ == "__main__":
    main()
