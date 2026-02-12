"""
DynamoDB → 로컬 DB 동기화

Lambda 크롤러가 DynamoDB에 저장한 영상 정보를 로컬 SQLite DB로 동기화합니다.

사용법:
    python lambda/dynamodb_sync.py
    python lambda/dynamodb_sync.py --limit 1000 --mark-collected
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================
# 설정
# ============================================================

@dataclass
class SyncConfig:
    """동기화 설정"""
    dynamodb_table: str = os.environ.get("DYNAMODB_TABLE", "robot-videos")
    region: str = os.environ.get("AWS_REGION", "ap-northeast-2")
    local_db_path: str = str(PROJECT_ROOT / "data" / "pade.db")
    batch_size: int = 100
    mark_collected: bool = True
    limit: Optional[int] = None  # None = 전부
    only_uncollected: bool = True


# ============================================================
# DynamoDB 스캐너
# ============================================================

class DynamoDBScanner:
    """DynamoDB 테이블 스캔 및 관리"""

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self._dynamodb = boto3.resource(
            "dynamodb",
            region_name=self.config.region,
        )
        self._table = self._dynamodb.Table(self.config.dynamodb_table)

    def scan_uncollected(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        아직 수집되지 않은 아이템 스캔

        Args:
            limit: 최대 스캔 수

        Returns:
            DynamoDB 아이템 리스트
        """
        items = []
        scan_kwargs = {}

        if self.config.only_uncollected:
            scan_kwargs["FilterExpression"] = boto3.dynamodb.conditions.Attr("collected").eq(False)

        max_items = limit or self.config.limit

        while True:
            response = self._table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))

            if max_items and len(items) >= max_items:
                items = items[:max_items]
                break

            # 페이지네이션
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        logger.info(f"DynamoDB 스캔 완료: {len(items)}개 아이템")
        return items

    def scan_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """전체 아이템 스캔"""
        items = []
        scan_kwargs = {}
        max_items = limit or self.config.limit

        while True:
            response = self._table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))

            if max_items and len(items) >= max_items:
                items = items[:max_items]
                break

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        return items

    def mark_collected(self, video_ids: List[str]):
        """
        DynamoDB에서 collected=True 마킹

        Args:
            video_ids: 마킹할 video_id 리스트
        """
        marked = 0
        for vid in video_ids:
            try:
                self._table.update_item(
                    Key={"video_id": vid},
                    UpdateExpression="SET collected = :val, collected_at = :ts",
                    ExpressionAttributeValues={
                        ":val": True,
                        ":ts": datetime.now().isoformat(),
                    },
                )
                marked += 1
            except Exception as e:
                logger.warning(f"마킹 실패 {vid}: {e}")

        logger.info(f"DynamoDB 마킹 완료: {marked}/{len(video_ids)}")

    def get_stats(self) -> Dict[str, Any]:
        """테이블 통계"""
        try:
            response = self._table.scan(Select="COUNT")
            total = response.get("Count", 0)

            collected_resp = self._table.scan(
                Select="COUNT",
                FilterExpression=boto3.dynamodb.conditions.Attr("collected").eq(True),
            )
            collected = collected_resp.get("Count", 0)

            return {
                "total": total,
                "collected": collected,
                "uncollected": total - collected,
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# 로컬 DB 동기화
# ============================================================

class LocalDBSync:
    """DynamoDB → 로컬 SQLite 동기화"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(PROJECT_ROOT / "data" / "pade.db")
        self._conn = None

    def _get_connection(self):
        """SQLite 연결"""
        if self._conn is None:
            import sqlite3
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._ensure_table()
        return self._conn

    def _ensure_table(self):
        """videos 테이블 존재 보장"""
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                title TEXT,
                url TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'pending',
                keyword TEXT,
                platform TEXT DEFAULT 'youtube',
                quality_score REAL,
                file_size INTEGER,
                duration INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                synced_from TEXT DEFAULT 'dynamodb'
            )
        """)
        conn.commit()

    def sync_items(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        DynamoDB 아이템을 로컬 DB에 동기화

        Args:
            items: DynamoDB 아이템 리스트

        Returns:
            {"inserted": N, "skipped": N, "errors": N}
        """
        conn = self._get_connection()
        inserted = 0
        skipped = 0
        errors = 0

        for item in items:
            vid = item.get("video_id", "")
            if not vid:
                errors += 1
                continue

            try:
                # 중복 확인
                cur = conn.execute(
                    "SELECT 1 FROM videos WHERE video_id = ?", (vid,)
                )
                if cur.fetchone():
                    skipped += 1
                    continue

                # 메타데이터 파싱
                metadata = item.get("metadata", "{}")
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata, ensure_ascii=False)

                # 삽입
                conn.execute(
                    """
                    INSERT INTO videos (video_id, title, url, metadata, keyword, platform, synced_from)
                    VALUES (?, ?, ?, ?, ?, ?, 'dynamodb')
                    """,
                    (
                        vid,
                        item.get("title", ""),
                        item.get("url", ""),
                        metadata,
                        item.get("keyword", ""),
                        item.get("platform", "youtube"),
                    ),
                )
                inserted += 1

            except Exception as e:
                logger.warning(f"동기화 실패 {vid}: {e}")
                errors += 1

        conn.commit()

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        logger.info(
            f"로컬 DB 동기화 완료: {inserted}개 삽입, "
            f"{skipped}개 중복, {errors}개 에러"
        )
        return result

    def get_local_count(self) -> int:
        """로컬 DB 비디오 수"""
        conn = self._get_connection()
        cur = conn.execute("SELECT COUNT(*) FROM videos")
        return cur.fetchone()[0]

    def close(self):
        """연결 종료"""
        if self._conn:
            self._conn.close()
            self._conn = None


# ============================================================
# 전체 동기화 프로세스
# ============================================================

def run_sync(config: Optional[SyncConfig] = None) -> Dict[str, Any]:
    """
    전체 동기화 실행: DynamoDB 스캔 → 로컬 DB 삽입 → collected 마킹

    Args:
        config: 동기화 설정

    Returns:
        동기화 결과 요약
    """
    config = config or SyncConfig()
    start_time = time.time()

    logger.info("🔄 DynamoDB → 로컬 DB 동기화 시작")

    # 1. DynamoDB 스캔
    scanner = DynamoDBScanner(config)
    items = scanner.scan_uncollected(limit=config.limit)

    if not items:
        logger.info("동기화할 새 아이템 없음")
        return {"items_scanned": 0, "inserted": 0, "skipped": 0}

    # 2. 로컬 DB 동기화
    local_sync = LocalDBSync(db_path=config.local_db_path)
    sync_result = local_sync.sync_items(items)

    # 3. DynamoDB에서 collected 마킹
    if config.mark_collected and sync_result["inserted"] > 0:
        synced_ids = [
            item["video_id"] for item in items
            if item.get("video_id")
        ]
        scanner.mark_collected(synced_ids)

    local_sync.close()
    elapsed = round(time.time() - start_time, 2)

    summary = {
        "items_scanned": len(items),
        **sync_result,
        "elapsed_sec": elapsed,
    }

    logger.info(f"✅ 동기화 완료: {summary}")
    return summary


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DynamoDB → 로컬 DB 동기화")
    parser.add_argument("--table", default=os.environ.get("DYNAMODB_TABLE", "robot-videos"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-northeast-2"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mark-collected", action="store_true", default=True)
    parser.add_argument("--no-mark", dest="mark_collected", action="store_false")
    parser.add_argument("--stats", action="store_true", help="통계만 출력")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    config = SyncConfig(
        dynamodb_table=args.table,
        region=args.region,
        limit=args.limit,
        mark_collected=args.mark_collected,
    )
    if args.db_path:
        config.local_db_path = args.db_path

    if args.stats:
        scanner = DynamoDBScanner(config)
        stats = scanner.get_stats()
        print(json.dumps(stats, indent=2))
        return

    result = run_sync(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
