"""
데이터베이스 연결 및 세션 관리
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Generator

from config import config
from models.database import Base


# 데이터베이스 엔진 생성
engine = create_engine(
    config.DATABASE_URL,
    poolclass=NullPool if config.TESTING else None,
    echo=config.DEBUG
)

# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """데이터베이스 삭제 (테스트용)"""
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    데이터베이스 세션 컨텍스트 매니저
    
    사용 예:
        with get_db() as db:
            videos = db.query(Video).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
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
