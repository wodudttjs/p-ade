"""
데이터베이스 연결 및 세션 관리
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError, DisconnectionError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from config import config
from models.database import Base

logger = logging.getLogger(__name__)

# ─── 엔진 생성 ───────────────────────────────────────────────────────────────

def _make_engine():
    if getattr(config, "TESTING", False):
        return create_engine(
            config.DATABASE_URL,
            poolclass=NullPool,
            echo=getattr(config, "DEBUG", False),
        )
    return create_engine(
        config.DATABASE_URL,
        poolclass=QueuePool,
        pool_size=20,          # 기본 연결 20개
        max_overflow=30,       # 최대 50개 연결
        pool_timeout=30,       # 연결 대기 최대 30초
        pool_recycle=3600,     # 1시간마다 재연결
        pool_pre_ping=True,    # 연결 끊김 자동 감지
        echo=getattr(config, "DEBUG", False),
    )


engine = _make_engine()


@event.listens_for(engine, "connect")
def _on_connect(dbapi_connection, connection_record):
    logger.debug("DB 연결 생성")


@event.listens_for(engine, "checkout")
def _on_checkout(dbapi_connection, connection_record, connection_proxy):
    """체크아웃 시 연결 상태 검증 — 끊어진 연결은 자동 재연결"""
    try:
        dbapi_connection.cursor().execute("SELECT 1")
    except Exception:
        raise DisconnectionError("DB 연결 끊김 — 재연결 중")


# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ─── 초기화 ───────────────────────────────────────────────────────────────────

def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """데이터베이스 삭제 (테스트용)"""
    Base.metadata.drop_all(bind=engine)


# ─── 세션 컨텍스트 매니저 ─────────────────────────────────────────────────────

@contextmanager
def get_db(
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Generator[Session, None, None]:
    """
    데이터베이스 세션 컨텍스트 매니저 (자동 commit/rollback, 데드락 재시도)

    사용 예:
        with get_db() as db:
            videos = db.query(Video).all()
    """
    for attempt in range(1, max_retries + 1):
        db = SessionLocal()
        try:
            yield db
            db.commit()
            return
        except OperationalError as e:
            db.rollback()
            if attempt < max_retries:
                logger.warning(
                    f"DB OperationalError (시도 {attempt}/{max_retries}): {e}. "
                    f"{retry_delay:.1f}초 후 재시도..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"DB 연결 실패 — 최대 재시도 초과: {e}")
                raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def get_db_session() -> Session:
    """
    데이터베이스 세션 반환 (수동 관리)

    주의: 사용 후 반드시 close() 호출 필요
    """
    return SessionLocal()


def check_db_connection() -> bool:
    """DB 연결 상태 확인"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB 연결 확인 실패: {e}")
        return False
