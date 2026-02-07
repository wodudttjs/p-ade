"""
데이터 서비스

데이터베이스에서 대시보드용 데이터를 조회하는 서비스 레이어
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from sqlalchemy import create_engine, func, text, and_, or_
from sqlalchemy.orm import sessionmaker, Session

# 순환 import 방지를 위해 직접 import
import sys
import os

# 프로젝트 루트를 sys.path에 추가 (models.database import를 위해)
_dashboard_dir = os.path.dirname(__file__)
_project_root = os.path.dirname(_dashboard_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# models.py 직접 import (dashboard 패키지 전체 로드 방지)
import importlib.util
_models_path = os.path.join(_dashboard_dir, "models.py")
_spec = importlib.util.spec_from_file_location("dashboard_models", _models_path)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)

JobRow = _models.JobRow
KPIData = _models.KPIData
QualityStats = _models.QualityStats
SystemStats = _models.SystemStats
STAGES = _models.STAGES
STATUSES = _models.STATUSES

# 로깅 설정 (loguru 없으면 기본 logging 사용)
try:
    from core.logging_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DataService:
    """
    대시보드 데이터 서비스
    
    실제 DB에서 데이터를 조회하여 대시보드 모델로 변환
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        DataService 초기화
        
        Args:
            db_url: 데이터베이스 URL. None이면 config에서 로드
        """
        self._engine = None
        self._session_factory = None
        self._db_url = db_url
        
    def _get_db_url(self) -> str:
        """DB URL 가져오기"""
        if self._db_url:
            return self._db_url

        env_db_url = os.getenv("P_ADE_DB_URL") or os.getenv("DASHBOARD_DB_URL")
        if env_db_url:
            return env_db_url

        env_db_path = os.getenv("P_ADE_DB_PATH")
        if env_db_path:
            return f"sqlite:///{env_db_path}"
        
        try:
            from config.settings import Config
            config = Config()
            # PostgreSQL 연결 테스트
            db_url = config.DATABASE_URL
            test_engine = create_engine(db_url)
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return db_url
        except Exception:
            # PostgreSQL 실패 시 SQLite fallback
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sqlite_path = os.path.join(base_dir, "data", "pade.db")
            return f"sqlite:///{sqlite_path}"
    
    def _get_engine(self):
        """SQLAlchemy 엔진 (lazy loading)"""
        if self._engine is None:
            db_url = self._get_db_url()
            self._engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
        return self._engine
    
    def _get_session(self) -> Session:
        """세션 생성"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self._get_engine())
        return self._session_factory()
    
    def is_connected(self) -> bool:
        """DB 연결 확인"""
        try:
            with self._get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"DB connection check failed: {e}")
            return False

    def get_db_status(self) -> str:
        return "connected" if self.is_connected() else "disconnected"
    
    def get_s3_status(self) -> str:
        try:
            from config.settings import Config
            import boto3

            config = Config()
            bucket = os.getenv("AWS_S3_BUCKET", config.AWS_S3_BUCKET)
            region = os.getenv("AWS_REGION", config.AWS_REGION)
            access_key = os.getenv("AWS_ACCESS_KEY_ID", config.AWS_ACCESS_KEY_ID)
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", config.AWS_SECRET_ACCESS_KEY)

            if not access_key or not secret_key:
                return "no-credentials"

            client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            client.head_bucket(Bucket=bucket)
            return "connected"
        except KeyboardInterrupt:
            return "disconnected"
        except Exception as e:
            logger.warning(f"S3 status check failed: {e}")
            return "disconnected"
    
    # ===== Jobs 관련 =====
    
    def get_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[JobRow]:
        """
        작업 목록 조회
        
        Args:
            limit: 최대 결과 수
            offset: 오프셋
            stage: 스테이지 필터
            status: 상태 필터
            query: 검색어 (job_key, video_id 등)
        
        Returns:
            JobRow 리스트
        """
        try:
            from models.database import ProcessingJob
            
            session = self._get_session()
            try:
                q = session.query(ProcessingJob)
                
                # 필터링
                if stage and stage != "all":
                    q = q.filter(ProcessingJob.stage == stage)
                if status and status != "all":
                    q = q.filter(ProcessingJob.status == status)
                if query:
                    q = q.filter(
                        or_(
                            ProcessingJob.job_key.ilike(f"%{query}%"),
                            ProcessingJob.video_id.ilike(f"%{query}%"),
                            ProcessingJob.failure_reason.ilike(f"%{query}%"),
                        )
                    )
                
                # 정렬 및 페이징
                q = q.order_by(ProcessingJob.created_at.desc())
                q = q.offset(offset).limit(limit)
                
                jobs = q.all()
                
                return [
                    JobRow(
                        job_key=job.job_key,
                        run_id=job.processing_version or "unknown",
                        stage=job.stage or "unknown",
                        status=job.status,
                        error_type=job.failure_type,
                        started_at=job.started_at or job.created_at,
                        duration_ms=self._calc_duration_ms(job.started_at, job.completed_at),
                        video_id=job.video_id,
                        episode_id=None,  # ProcessingJob에는 episode_id 없음
                        retry_count=job.retry_count,
                        log_snippet=job.failure_reason or "",
                    )
                    for job in jobs
                ]
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to get jobs: {e}")
            return []
    
    def get_job_stats(self) -> Dict[str, int]:
        """작업 상태별 통계"""
        try:
            from models.database import ProcessingJob
            
            session = self._get_session()
            try:
                stats = {}
                
                # 전체 개수
                stats["total"] = session.query(ProcessingJob).count()
                
                # 상태별 개수
                for status in ["pending", "running", "completed", "failed", "skipped"]:
                    count = session.query(ProcessingJob).filter(
                        ProcessingJob.status == status
                    ).count()
                    # 대시보드 모델과 매핑
                    if status == "completed":
                        stats["success"] = count
                    elif status == "failed":
                        stats["fail"] = count
                    elif status == "skipped":
                        stats["skip"] = count
                    else:
                        stats[status] = count
                
                return stats
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to get job stats: {e}")
            return {"total": 0}
    
    # ===== KPI 관련 =====
    
    def get_kpi(self) -> KPIData:
        """KPI 데이터 조회"""
        try:
            from models.database import Video, Episode, ProcessingJob, StorageCost
            
            session = self._get_session()
            try:
                # 비디오 통계
                total_videos = session.query(Video).count()
                downloaded = session.query(Video).filter(
                    Video.downloaded_at.isnot(None)
                ).count()
                
                # 에피소드 통계
                episodes = session.query(Episode).count()
                high_quality = session.query(Episode).filter(
                    Episode.quality_score >= 0.8
                ).count()
                
                # 처리 통계
                job_stats = self.get_job_stats()
                total_jobs = job_stats.get("total", 0)
                success_jobs = job_stats.get("success", 0)
                success_rate = success_jobs / max(total_jobs, 1)
                
                running_jobs = job_stats.get("running", 0)
                pending_jobs = job_stats.get("pending", 0)
                
                # 평균 처리 시간
                avg_time_result = session.query(
                    func.avg(
                        func.extract('epoch', ProcessingJob.completed_at) -
                        func.extract('epoch', ProcessingJob.started_at)
                    )
                ).filter(
                    ProcessingJob.status == "completed",
                    ProcessingJob.started_at.isnot(None),
                    ProcessingJob.completed_at.isnot(None),
                ).scalar()
                avg_processing_time = avg_time_result or 0
                
                # 스토리지 (최신 기록)
                latest_cost = session.query(StorageCost).order_by(
                    StorageCost.period_end.desc()
                ).first()
                
                storage_gb = 0.0
                monthly_cost = 0.0
                if latest_cost:
                    storage_gb = latest_cost.total_bytes / (1024**3)
                    monthly_cost = latest_cost.total_cost
                
                return KPIData(
                    total_videos=total_videos,
                    downloaded=downloaded,
                    episodes=episodes,
                    high_quality=high_quality,
                    storage_gb=storage_gb,
                    monthly_cost=monthly_cost,
                    success_rate=success_rate,
                    avg_processing_time_sec=avg_processing_time,
                    queue_depth=pending_jobs,
                    active_workers=running_jobs,
                )
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to get KPI: {e}")
            return KPIData()
    
    # ===== Quality 관련 =====
    
    def get_quality_stats(self) -> QualityStats:
        """품질 통계 조회"""
        try:
            from models.database import Episode
            
            session = self._get_session()
            try:
                total_episodes = session.query(Episode).count()
                
                if total_episodes == 0:
                    return QualityStats()
                
                # 품질 통과 (quality_score >= 0.7)
                passed = session.query(Episode).filter(
                    Episode.quality_score >= 0.7
                ).count()
                failed = total_episodes - passed
                pass_rate = passed / max(total_episodes, 1)
                
                # 통계 (SQLite 호환 - stddev, percentile_cont 사용 안 함)
                stats = session.query(
                    func.avg(Episode.confidence_score).label("conf_mean"),
                    func.avg(Episode.jittering_score).label("jitter_mean"),
                    func.avg(Episode.duration_frames).label("length_mean"),
                ).first()
                
                # stddev 대신 수동 계산 (SQLite 호환)
                conf_std = 0.0
                jitter_p95 = 0.0
                try:
                    all_conf = [e.confidence_score for e in session.query(Episode.confidence_score).all() if e.confidence_score]
                    if all_conf:
                        import statistics
                        conf_std = statistics.stdev(all_conf) if len(all_conf) > 1 else 0.0
                    
                    all_jitter = sorted([e.jittering_score for e in session.query(Episode.jittering_score).all() if e.jittering_score])
                    if all_jitter:
                        idx = int(len(all_jitter) * 0.95)
                        jitter_p95 = all_jitter[min(idx, len(all_jitter) - 1)]
                except Exception:
                    pass
                
                return QualityStats(
                    total_episodes=total_episodes,
                    passed=passed,
                    failed=failed,
                    pass_rate=pass_rate,
                    confidence_mean=stats.conf_mean or 0.0,
                    confidence_std=conf_std,
                    jitter_mean=stats.jitter_mean or 0.0,
                    jitter_std=0.0,  # TODO: calculate from data
                    jitter_p95=jitter_p95,
                    length_mean=stats.length_mean or 0.0,
                    episode_length_mean=stats.length_mean or 0.0,
                )
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to get quality stats: {e}")
            return QualityStats()
    
    # ===== System 관련 =====
    
    def get_system_stats(self) -> SystemStats:
        """시스템 리소스 통계"""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            # GPU 정보 (있으면)
            gpu_util = 0.0
            gpu_mem = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
                    gpu_mem = gpus[0].memoryUtil * 100
            except ImportError:
                pass
            
            return SystemStats(
                cpu_percent=cpu,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                gpu_util_percent=gpu_util,
                gpu_memory_percent=gpu_mem,
            )
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return SystemStats()
    
    # ===== 유틸리티 =====
    
    def _calc_duration_ms(
        self,
        started_at: Optional[datetime],
        completed_at: Optional[datetime],
    ) -> Optional[int]:
        """처리 시간 계산 (ms)"""
        if not started_at:
            return None
        
        end = completed_at or datetime.utcnow()
        delta = end - started_at
        return int(delta.total_seconds() * 1000)


# 싱글톤 인스턴스
_data_service: Optional[DataService] = None


def get_data_service(db_url: Optional[str] = None) -> DataService:
    """DataService 싱글톤 인스턴스"""
    global _data_service
    if _data_service is None:
        _data_service = DataService(db_url)
    return _data_service
