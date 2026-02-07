"""
데이터 모델 및 Mock 데이터

Dashboard에서 사용하는 데이터 구조
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum


class Stage(Enum):
    """파이프라인 스테이지"""
    DISCOVER = "discover"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    UPLOAD = "upload"
    FINALIZE = "finalize"


class JobStatus(Enum):
    """작업 상태"""
    RUNNING = "running"
    SUCCESS = "success"
    SKIP = "skip"
    FAIL = "fail"
    PENDING = "pending"


class ErrorType(Enum):
    """에러 유형"""
    NETWORK_TRANSIENT = "NETWORK_TRANSIENT"
    RATE_LIMIT = "RATE_LIMIT"
    INTEGRITY_FAIL = "INTEGRITY_FAIL"
    QUALITY_SKIP = "QUALITY_SKIP"
    AUTHORIZATION = "AUTHORIZATION"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    TIMEOUT = "TIMEOUT"
    SYSTEM_ERROR = "SYSTEM_ERROR"


STAGES = [s.value for s in Stage]
STATUSES = [s.value for s in JobStatus if s != JobStatus.PENDING]
ERROR_TYPES = [e.value for e in ErrorType]


@dataclass
class JobRow:
    """작업 행 데이터"""
    job_key: str
    run_id: str
    stage: str
    status: str
    error_type: Optional[str]
    started_at: datetime
    duration_ms: Optional[int]
    video_id: Optional[str]
    episode_id: Optional[str]
    retry_count: int = 0
    log_snippet: str = ""


@dataclass
class KPIData:
    """KPI 대시보드 데이터"""
    total_videos: int = 0
    downloaded: int = 0
    episodes: int = 0
    high_quality: int = 0
    storage_gb: float = 0.0
    monthly_cost: float = 0.0
    
    # 추가 메트릭
    success_rate: float = 0.0
    avg_processing_time_sec: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    
    # 품질 분포
    confidence_mean: float = 0.0
    jitter_mean: float = 0.0


@dataclass
class QualityStats:
    """품질 통계"""
    total_episodes: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    jitter_mean: float = 0.0
    jitter_std: float = 0.0
    jitter_p95: float = 0.0
    length_mean: float = 0.0  # alias for CLI
    
    episode_length_mean: float = 0.0
    nan_ratio_mean: float = 0.0


@dataclass
class SystemStats:
    """시스템 리소스 통계"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    
    gpu_util_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_gb: float = 0.0


def make_mock_jobs(n: int = 80) -> List[JobRow]:
    """Mock 작업 데이터 생성"""
    rows = []
    now = datetime.now()
    
    log_samples = [
        "[INFO] Starting download...\n[INFO] Format: 1080p mp4\n[INFO] Downloading...",
        "[INFO] Extracting pose...\n[INFO] MediaPipe initialized\n[INFO] Processing frames...",
        "[ERROR] Connection timeout\n[WARN] Retrying (1/3)...\n[ERROR] Failed after 3 retries",
        "[INFO] Upload started\n[INFO] S3 bucket: p-ade-data\n[INFO] Uploaded successfully",
        "[INFO] Quality check passed\n[INFO] Confidence: 0.87\n[INFO] Jitter: 0.12",
    ]
    
    for i in range(n):
        stage = STAGES[i % 5]
        status = random.choice(STATUSES)
        err = random.choice(ERROR_TYPES) if status == "fail" else None
        started = now - timedelta(minutes=i * 4)
        dur = None if status == "running" else random.randint(1200, 320000)
        vid = f"vid_{1000+i}"
        ep = f"{vid}_ep{str((i % 7) + 1).zfill(3)}" if stage in ("transform", "upload") else None
        
        rows.append(JobRow(
            job_key=f"{stage}:youtube:{vid}:v4.0.0",
            run_id=f"run_{i//10}",
            stage=stage,
            status=status,
            error_type=err,
            started_at=started,
            duration_ms=dur,
            video_id=vid,
            episode_id=ep,
            retry_count=random.randint(0, 3) if status == "fail" else 0,
            log_snippet=random.choice(log_samples),
        ))
    
    return rows


def make_mock_kpi() -> KPIData:
    """Mock KPI 데이터 생성"""
    return KPIData(
        total_videos=random.randint(500, 600),
        downloaded=random.randint(450, 550),
        episodes=random.randint(12000, 15000),
        high_quality=random.randint(6000, 8000),
        storage_gb=round(random.uniform(35, 50), 1),
        monthly_cost=round(random.uniform(8, 15), 2),
        success_rate=round(random.uniform(0.85, 0.95), 2),
        avg_processing_time_sec=round(random.uniform(30, 60), 1),
        queue_depth=random.randint(0, 50),
        active_workers=random.randint(1, 4),
        confidence_mean=round(random.uniform(0.75, 0.90), 2),
        jitter_mean=round(random.uniform(0.05, 0.15), 3),
    )


def make_mock_quality() -> QualityStats:
    """Mock 품질 통계 생성"""
    total = random.randint(1000, 2000)
    passed = int(total * random.uniform(0.7, 0.9))
    
    return QualityStats(
        total_episodes=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(passed / total, 2),
        confidence_mean=round(random.uniform(0.75, 0.90), 2),
        confidence_std=round(random.uniform(0.05, 0.15), 3),
        jitter_mean=round(random.uniform(0.05, 0.15), 3),
        jitter_p95=round(random.uniform(0.15, 0.30), 3),
        episode_length_mean=round(random.uniform(80, 150), 1),
        nan_ratio_mean=round(random.uniform(0.01, 0.05), 3),
    )


def make_mock_system() -> SystemStats:
    """Mock 시스템 통계 생성"""
    return SystemStats(
        cpu_percent=round(random.uniform(20, 80), 1),
        memory_percent=round(random.uniform(40, 70), 1),
        memory_used_gb=round(random.uniform(8, 16), 1),
        disk_percent=round(random.uniform(30, 60), 1),
        disk_used_gb=round(random.uniform(100, 300), 1),
        gpu_util_percent=round(random.uniform(0, 90), 1),
        gpu_memory_gb=round(random.uniform(2, 8), 1),
    )
