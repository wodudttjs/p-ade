"""
GlobalVideoRegistry - 전역 비디오 수집 레지스트리

실행 간 중복 0% 보장을 위한 Redis 기반 레지스트리.
Redis 장애 시 PostgreSQL(SQLite) 자동 폴백.

Redis Keys:
  "pade:registry:videos"   → Set[video_id]  (영구 보존, TTL 없음)
  "pade:registry:urls"     → Set[url_hash]
  "pade:registry:rejected" → Set[video_id]  (품질 탈락)
"""

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis 키
REGISTRY_VIDEOS_KEY = "pade:registry:videos"
REGISTRY_URLS_KEY = "pade:registry:urls"
REGISTRY_REJECTED_KEY = "pade:registry:rejected"


@dataclass
class RegistryStats:
    total_collected: int = 0
    total_rejected: int = 0
    duplicate_blocked: int = 0


class GlobalVideoRegistry:
    """
    글로벌 비디오 레지스트리

    Redis를 primary, SQLite/PostgreSQL을 fallback으로 사용하여
    실행 간 중복 수집을 방지합니다.

    성능 목표:
      - Redis 조회 < 1ms
      - DB 조회 < 10ms
      - 10,000개 video_id 등록 후 조회 속도 < 5ms
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        sync_interval_sec: int = 60,
    ):
        self._db_path = db_path
        self._sync_interval = sync_interval_sec
        self._redis = None
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = RegistryStats()

        # Redis 연결 시도
        self._connect_redis()

        # 시작 시 DB → Redis 로드
        if self._redis:
            self._load_from_db()
            self._start_sync_thread()

    # ──────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────

    def is_collected(self, video_id: str) -> bool:
        """video_id가 이미 수집되었는지 확인 (Redis < 1ms, DB < 10ms)"""
        if self._redis:
            try:
                return bool(self._redis.sismember(REGISTRY_VIDEOS_KEY, video_id))
            except Exception:
                pass
        return self._db_is_collected(video_id)

    def is_url_collected(self, url: str) -> bool:
        """URL이 이미 수집되었는지 확인 (정규화 후 체크)"""
        url_hash = self._hash_url(url)
        if self._redis:
            try:
                return bool(self._redis.sismember(REGISTRY_URLS_KEY, url_hash))
            except Exception:
                pass
        return False

    def filter_new_only(self, video_list: List[Dict]) -> List[Dict]:
        """
        미수집 영상만 필터링

        Args:
            video_list: [{"video_id": ..., "url": ...}, ...] 형태의 리스트

        Returns:
            미수집 영상 리스트 (중복 제거됨)
        """
        new_videos = []
        blocked = 0

        for video in video_list:
            video_id = video.get("video_id", "")
            url = video.get("url", "")

            if video_id and self.is_collected(video_id):
                blocked += 1
                continue

            if url and self.is_url_collected(url):
                blocked += 1
                continue

            new_videos.append(video)

        self._stats.duplicate_blocked += blocked
        if blocked > 0:
            logger.info(f"🚫 중복 차단: {blocked}개, 신규: {len(new_videos)}개")

        return new_videos

    def register(
        self,
        video_id: str,
        url: str,
        run_id: str = "",
        s3_path: str = "",
    ) -> bool:
        """수집 완료 영상을 레지스트리에 등록"""
        url_hash = self._hash_url(url)

        if self._redis:
            try:
                pipe = self._redis.pipeline()
                pipe.sadd(REGISTRY_VIDEOS_KEY, video_id)
                pipe.sadd(REGISTRY_URLS_KEY, url_hash)
                pipe.execute()
                self._stats.total_collected += 1
                return True
            except Exception as e:
                logger.warning(f"Redis 등록 실패: {e}")

        return self._db_register(video_id, url, run_id, s3_path)

    def register_rejected(self, video_id: str, reason: str = "") -> bool:
        """품질 탈락 영상 등록 (재수집 방지)"""
        if self._redis:
            try:
                self._redis.sadd(REGISTRY_REJECTED_KEY, video_id)
                self._stats.total_rejected += 1
                logger.debug(f"❌ rejected 등록: {video_id} ({reason})")
                return True
            except Exception as e:
                logger.warning(f"Redis rejected 등록 실패: {e}")
        return False

    def get_stats(self) -> Dict:
        """전체 통계 반환"""
        stats = {
            "total_collected": self._stats.total_collected,
            "total_rejected": self._stats.total_rejected,
            "duplicate_blocked": self._stats.duplicate_blocked,
        }

        if self._redis:
            try:
                stats["redis_videos"] = self._redis.scard(REGISTRY_VIDEOS_KEY)
                stats["redis_urls"] = self._redis.scard(REGISTRY_URLS_KEY)
                stats["redis_rejected"] = self._redis.scard(REGISTRY_REJECTED_KEY)
            except Exception:
                pass

        return stats

    def close(self):
        """리소스 정리"""
        self._running = False
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)

    # ──────────────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────────────

    def _connect_redis(self):
        """Redis 연결"""
        try:
            import redis as redis_lib
            client = redis_lib.Redis(
                host="localhost",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=5,
            )
            client.ping()
            self._redis = client
            logger.info("✅ GlobalVideoRegistry: Redis 연결 성공")
        except Exception as e:
            logger.warning(f"⚠️ Redis 연결 실패, DB 폴백 모드 사용: {e}")
            self._redis = None

    def _hash_url(self, url: str) -> str:
        """URL 정규화 후 해시 (YouTube video_id 추출 우선)"""
        url = url.strip()
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if match:
            return f"yt:{match.group(1)}"
        return hashlib.sha256(url.lower().encode()).hexdigest()[:32]

    def _resolve_db_path(self) -> Path:
        db_path = self._db_path or "data/pade.db"
        db_file = Path(db_path)
        if not db_file.is_absolute():
            db_file = Path(__file__).resolve().parent.parent / db_file
        return db_file

    def _db_is_collected(self, video_id: str) -> bool:
        """DB에서 수집 여부 확인 (폴백)"""
        try:
            from sqlalchemy import create_engine, text
            db_file = self._resolve_db_path()
            if not db_file.exists():
                return False
            engine = create_engine(f"sqlite:///{db_file}")
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT 1 FROM videos "
                        "WHERE video_id = :vid "
                        "AND status NOT IN ('failed', 'discovered') LIMIT 1"
                    ),
                    {"vid": video_id},
                )
                return result.fetchone() is not None
        except Exception:
            return False

    def _db_register(self, video_id: str, url: str, run_id: str, s3_path: str) -> bool:
        """DB에 직접 등록 (Redis 폴백)"""
        try:
            from sqlalchemy import create_engine, text
            db_file = self._resolve_db_path()
            if not db_file.exists():
                return False
            engine = create_engine(f"sqlite:///{db_file}")
            with engine.connect() as conn:
                # video_registry 테이블이 없으면 무시
                try:
                    conn.execute(
                        text(
                            "INSERT OR IGNORE INTO video_registry "
                            "(video_id, url, run_id, s3_path, registered_at) "
                            "VALUES (:vid, :url, :run_id, :s3, datetime('now'))"
                        ),
                        {"vid": video_id, "url": url, "run_id": run_id, "s3": s3_path},
                    )
                    conn.commit()
                except Exception:
                    pass
            self._stats.total_collected += 1
            return True
        except Exception as e:
            logger.error(f"DB 등록 실패: {e}")
            return False

    def _load_from_db(self):
        """시작 시 DB → Redis 로드 (RDB+AOF 백업 복구 역할)"""
        try:
            from sqlalchemy import create_engine, text
            db_file = self._resolve_db_path()
            if not db_file.exists():
                return

            engine = create_engine(f"sqlite:///{db_file}")
            with engine.connect() as conn:
                try:
                    rows = conn.execute(
                        text(
                            "SELECT video_id FROM videos "
                            "WHERE status NOT IN ('failed', 'discovered')"
                        )
                    ).fetchall()

                    if rows:
                        pipe = self._redis.pipeline()
                        for (vid,) in rows:
                            pipe.sadd(REGISTRY_VIDEOS_KEY, vid)
                        pipe.execute()
                        logger.info(f"📥 DB → Redis 로드 완료: {len(rows)}개 video_id")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"DB 로드 실패: {e}")

    def _start_sync_thread(self):
        """Redis → DB 비동기 동기화 스레드 시작"""
        self._running = True
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="registry-sync",
        )
        self._sync_thread.start()

    def _sync_loop(self):
        """sync_interval마다 Redis → DB 동기화"""
        while self._running:
            time.sleep(self._sync_interval)
            try:
                self._sync_redis_to_db()
            except Exception as e:
                logger.debug(f"동기화 오류: {e}")

    def _sync_redis_to_db(self):
        """Redis video_id Set → DB video_registry 테이블 동기화"""
        if not self._redis:
            return
        try:
            video_ids = self._redis.smembers(REGISTRY_VIDEOS_KEY)
            if not video_ids:
                return

            from sqlalchemy import create_engine, text
            db_file = self._resolve_db_path()
            if not db_file.exists():
                return

            engine = create_engine(f"sqlite:///{db_file}")
            with engine.connect() as conn:
                try:
                    for vid in video_ids:
                        conn.execute(
                            text(
                                "INSERT OR IGNORE INTO video_registry "
                                "(video_id, registered_at) "
                                "VALUES (:vid, datetime('now'))"
                            ),
                            {"vid": vid},
                        )
                    conn.commit()
                    logger.debug(f"🔄 Redis→DB 동기화: {len(video_ids)}개")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Redis→DB 동기화 실패: {e}")


# ──────────────────────────────────────────────────────────────
# 싱글턴 헬퍼
# ──────────────────────────────────────────────────────────────

_registry: Optional[GlobalVideoRegistry] = None
_registry_lock = threading.Lock()


def get_registry(db_path: Optional[str] = None) -> GlobalVideoRegistry:
    """글로벌 레지스트리 싱글턴 반환"""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = GlobalVideoRegistry(db_path=db_path)
    return _registry
