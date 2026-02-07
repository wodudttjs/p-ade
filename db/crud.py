"""
Database CRUD Operations

클라우드 파일 및 업로드 태스크 CRUD
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session

from core.logging_config import setup_logger
from models.database import (
    Base,
    CloudFile,
    UploadTask,
    StorageCost,
    DatasetVersion,
    Episode,
    Video,
)

logger = setup_logger(__name__)


class DatabaseManager:
    """데이터베이스 연결 및 세션 관리"""
    
    def __init__(self, database_url: str = "sqlite:///p-ade.db"):
        """
        DatabaseManager 초기화
        
        Args:
            database_url: 데이터베이스 연결 URL
                - SQLite: sqlite:///p-ade.db
                - PostgreSQL: postgresql://user:pass@host:5432/dbname
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        
    def create_tables(self):
        """테이블 생성"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
        
    def get_session(self) -> Session:
        """세션 생성"""
        return self.SessionLocal()
    
    def close(self):
        """연결 종료"""
        self.engine.dispose()


class CloudFileCRUD:
    """CloudFile CRUD 작업"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create(
        self,
        file_name: str,
        file_type: str,
        file_size_bytes: int,
        sha256: str,
        provider: str,
        bucket: str,
        key: str,
        uri: str,
        episode_id: Optional[int] = None,
        video_id: Optional[int] = None,
        dataset_version_id: Optional[int] = None,
        etag: Optional[str] = None,
        version_id: Optional[str] = None,
        storage_class: Optional[str] = None,
        compression: Optional[str] = None,
        original_size_bytes: Optional[int] = None,
        meta_data: Optional[Dict] = None,
        tags: Optional[Dict] = None,
    ) -> CloudFile:
        """CloudFile 생성"""
        file_id = str(uuid.uuid4())
        
        compression_ratio = None
        if compression and original_size_bytes and file_size_bytes > 0:
            compression_ratio = original_size_bytes / file_size_bytes
            
        cloud_file = CloudFile(
            file_id=file_id,
            episode_id=episode_id,
            video_id=video_id,
            dataset_version_id=dataset_version_id,
            file_name=file_name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            provider=provider,
            bucket=bucket,
            key=key,
            uri=uri,
            etag=etag,
            version_id=version_id,
            storage_class=storage_class,
            compression=compression,
            original_size_bytes=original_size_bytes,
            compression_ratio=compression_ratio,
            meta_data=meta_data or {},
            tags=tags or {},
        )
        
        self.db.add(cloud_file)
        self.db.commit()
        self.db.refresh(cloud_file)
        
        logger.debug(f"Created CloudFile: {file_id}")
        return cloud_file
    
    def get_by_id(self, file_id: str) -> Optional[CloudFile]:
        """file_id로 조회"""
        return self.db.query(CloudFile).filter(CloudFile.file_id == file_id).first()
    
    def get_by_sha256(self, sha256: str) -> Optional[CloudFile]:
        """SHA256으로 조회 (중복 체크)"""
        return self.db.query(CloudFile).filter(CloudFile.sha256 == sha256).first()
    
    def get_by_uri(self, uri: str) -> Optional[CloudFile]:
        """URI로 조회"""
        return self.db.query(CloudFile).filter(CloudFile.uri == uri).first()
    
    def get_by_episode(self, episode_id: int) -> List[CloudFile]:
        """에피소드의 모든 파일 조회"""
        return self.db.query(CloudFile).filter(CloudFile.episode_id == episode_id).all()
    
    def get_by_video(self, video_id: int) -> List[CloudFile]:
        """비디오의 모든 파일 조회"""
        return self.db.query(CloudFile).filter(CloudFile.video_id == video_id).all()
    
    def get_by_version(self, version_id: int) -> List[CloudFile]:
        """데이터셋 버전의 모든 파일 조회"""
        return self.db.query(CloudFile).filter(CloudFile.dataset_version_id == version_id).all()
    
    def list_all(
        self,
        provider: Optional[str] = None,
        bucket: Optional[str] = None,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CloudFile]:
        """파일 목록 조회"""
        query = self.db.query(CloudFile)
        
        if provider:
            query = query.filter(CloudFile.provider == provider)
        if bucket:
            query = query.filter(CloudFile.bucket == bucket)
        if file_type:
            query = query.filter(CloudFile.file_type == file_type)
        if status:
            query = query.filter(CloudFile.status == status)
            
        return query.order_by(desc(CloudFile.created_at)).offset(offset).limit(limit).all()
    
    def update_status(
        self,
        file_id: str,
        status: str,
        verified_at: Optional[datetime] = None,
    ) -> Optional[CloudFile]:
        """상태 업데이트"""
        cloud_file = self.get_by_id(file_id)
        if cloud_file:
            cloud_file.status = status
            if verified_at:
                cloud_file.verified_at = verified_at
            self.db.commit()
            self.db.refresh(cloud_file)
        return cloud_file
    
    def delete(self, file_id: str) -> bool:
        """삭제"""
        cloud_file = self.get_by_id(file_id)
        if cloud_file:
            self.db.delete(cloud_file)
            self.db.commit()
            return True
        return False
    
    def get_stats(
        self,
        provider: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """통계 조회"""
        from sqlalchemy import func
        
        query = self.db.query(
            func.count(CloudFile.id).label("total_files"),
            func.sum(CloudFile.file_size_bytes).label("total_bytes"),
            func.avg(CloudFile.compression_ratio).label("avg_compression_ratio"),
        )
        
        if provider:
            query = query.filter(CloudFile.provider == provider)
        if bucket:
            query = query.filter(CloudFile.bucket == bucket)
            
        result = query.first()
        
        return {
            "total_files": result.total_files or 0,
            "total_bytes": result.total_bytes or 0,
            "avg_compression_ratio": float(result.avg_compression_ratio or 0),
        }


class UploadTaskCRUD:
    """UploadTask CRUD 작업"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create(
        self,
        task_id: str,
        task_type: str,
        local_path: str,
        remote_key: str,
        bucket: str,
        provider: str,
        priority: int = 2,
        max_retries: int = 3,
    ) -> UploadTask:
        """UploadTask 생성"""
        task = UploadTask(
            task_id=task_id,
            task_type=task_type,
            local_path=local_path,
            remote_key=remote_key,
            bucket=bucket,
            provider=provider,
            priority=priority,
            max_retries=max_retries,
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        logger.debug(f"Created UploadTask: {task_id}")
        return task
    
    def get_by_id(self, task_id: str) -> Optional[UploadTask]:
        """task_id로 조회"""
        return self.db.query(UploadTask).filter(UploadTask.task_id == task_id).first()
    
    def update_status(
        self,
        task_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        cloud_file_id: Optional[int] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[UploadTask]:
        """상태 업데이트"""
        task = self.get_by_id(task_id)
        if task:
            task.status = status
            if started_at:
                task.started_at = started_at
            if completed_at:
                task.completed_at = completed_at
            if cloud_file_id:
                task.cloud_file_id = cloud_file_id
            if error_type:
                task.error_type = error_type
            if error_message:
                task.error_message = error_message
            self.db.commit()
            self.db.refresh(task)
        return task
    
    def increment_retry(self, task_id: str) -> Optional[UploadTask]:
        """재시도 횟수 증가"""
        task = self.get_by_id(task_id)
        if task:
            task.retry_count += 1
            task.status = "pending"
            self.db.commit()
            self.db.refresh(task)
        return task
    
    def get_pending(self, limit: int = 100) -> List[UploadTask]:
        """대기 중인 태스크 조회"""
        return (
            self.db.query(UploadTask)
            .filter(UploadTask.status == "pending")
            .filter(UploadTask.retry_count < UploadTask.max_retries)
            .order_by(UploadTask.priority, UploadTask.created_at)
            .limit(limit)
            .all()
        )
    
    def get_failed(self, limit: int = 100) -> List[UploadTask]:
        """실패한 태스크 조회"""
        return (
            self.db.query(UploadTask)
            .filter(UploadTask.status == "failed")
            .order_by(desc(UploadTask.completed_at))
            .limit(limit)
            .all()
        )
    
    def cleanup_old(self, days: int = 30) -> int:
        """오래된 완료 태스크 삭제"""
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        result = (
            self.db.query(UploadTask)
            .filter(UploadTask.status == "completed")
            .filter(UploadTask.completed_at < cutoff)
            .delete()
        )
        
        self.db.commit()
        return result


class StorageCostCRUD:
    """StorageCost CRUD 작업"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create(
        self,
        period_start: datetime,
        period_end: datetime,
        provider: str,
        bucket: Optional[str] = None,
        total_bytes: int = 0,
        storage_class: Optional[str] = None,
        storage_cost: float = 0.0,
        request_cost: float = 0.0,
        transfer_cost: float = 0.0,
        put_requests: int = 0,
        get_requests: int = 0,
        list_requests: int = 0,
        meta_data: Optional[Dict] = None,
    ) -> StorageCost:
        """StorageCost 생성"""
        cost = StorageCost(
            period_start=period_start,
            period_end=period_end,
            provider=provider,
            bucket=bucket,
            total_bytes=total_bytes,
            storage_class=storage_class,
            storage_cost=storage_cost,
            request_cost=request_cost,
            transfer_cost=transfer_cost,
            total_cost=storage_cost + request_cost + transfer_cost,
            put_requests=put_requests,
            get_requests=get_requests,
            list_requests=list_requests,
            meta_data=meta_data or {},
        )
        
        self.db.add(cost)
        self.db.commit()
        self.db.refresh(cost)
        
        return cost
    
    def get_by_period(
        self,
        start: datetime,
        end: datetime,
        provider: Optional[str] = None,
    ) -> List[StorageCost]:
        """기간별 비용 조회"""
        query = self.db.query(StorageCost).filter(
            and_(
                StorageCost.period_start >= start,
                StorageCost.period_end <= end,
            )
        )
        
        if provider:
            query = query.filter(StorageCost.provider == provider)
            
        return query.order_by(StorageCost.period_start).all()
    
    def get_total_cost(
        self,
        start: datetime,
        end: datetime,
        provider: Optional[str] = None,
    ) -> float:
        """기간별 총 비용"""
        from sqlalchemy import func
        
        query = self.db.query(func.sum(StorageCost.total_cost)).filter(
            and_(
                StorageCost.period_start >= start,
                StorageCost.period_end <= end,
            )
        )
        
        if provider:
            query = query.filter(StorageCost.provider == provider)
            
        result = query.scalar()
        return float(result or 0)


class DatasetVersionCRUD:
    """DatasetVersion CRUD 작업"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create(
        self,
        version: str,
        total_videos: int = 0,
        total_episodes: int = 0,
        total_size_bytes: int = 0,
        description: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> DatasetVersion:
        """DatasetVersion 생성"""
        dataset = DatasetVersion(
            version=version,
            total_videos=total_videos,
            total_episodes=total_episodes,
            total_size_bytes=total_size_bytes,
            description=description,
            manifest_path=manifest_path,
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        return dataset
    
    def get_by_version(self, version: str) -> Optional[DatasetVersion]:
        """버전으로 조회"""
        return self.db.query(DatasetVersion).filter(DatasetVersion.version == version).first()
    
    def get_latest(self) -> Optional[DatasetVersion]:
        """최신 버전 조회"""
        return (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.is_active == True)
            .order_by(desc(DatasetVersion.created_at))
            .first()
        )
    
    def list_all(self, active_only: bool = True) -> List[DatasetVersion]:
        """모든 버전 조회"""
        query = self.db.query(DatasetVersion)
        
        if active_only:
            query = query.filter(DatasetVersion.is_active == True)
            
        return query.order_by(desc(DatasetVersion.created_at)).all()
    
    def deactivate(self, version: str) -> Optional[DatasetVersion]:
        """버전 비활성화"""
        dataset = self.get_by_version(version)
        if dataset:
            dataset.is_active = False
            self.db.commit()
            self.db.refresh(dataset)
        return dataset
