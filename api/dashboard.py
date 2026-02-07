"""
Dashboard API 엔드포인트

UI 대시보드를 위한 REST API
- GET /api/overview - KPI 및 건강 상태
- GET /api/stages - 스테이지별 상태
- GET /api/jobs - 작업 목록
- GET /api/job/{job_key} - 작업 상세
- GET /api/quality/weekly - 주간 품질 리포트
- GET /api/versions - 데이터셋 버전
- GET /api/cost - 비용 추적
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import random
import math

app = FastAPI(
    title="P-ADE Dashboard API",
    description="Pipeline monitoring and analytics API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Enums =====

class StageName(str, Enum):
    DISCOVER = "discover"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    UPLOAD = "upload"
    FINALIZE = "finalize"


class JobStatus(str, Enum):
    SUCCESS = "success"
    FAIL = "fail"
    SKIP = "skip"
    RUNNING = "running"


class VersionStatus(str, Enum):
    DRAFT = "DRAFT"
    RELEASED = "RELEASED"
    DEPRECATED = "DEPRECATED"
    ROLLED_BACK = "ROLLED_BACK"


# ===== Response Models =====

class KPI(BaseModel):
    total_videos: int
    downloaded_videos: int
    total_episodes: int
    high_quality_episodes: int
    storage_gb: float
    monthly_cost_usd: float


class Health(BaseModel):
    error_rate_pct: float
    p95_end_to_end_ms: int
    queue_backlog: int
    last_alert: Optional[str] = None


class ThroughputPoint(BaseModel):
    ts: str
    jobs: int
    errors: int


class OverviewResponse(BaseModel):
    range: str
    kpi: KPI
    health: Health
    throughput: List[ThroughputPoint]


class StageStatus(BaseModel):
    stage: StageName
    success: int
    fail: int
    skip: int
    p95_ms: int
    inflight: int
    queue_depth: int


class JobRow(BaseModel):
    job_key: str
    run_id: str
    stage: StageName
    status: JobStatus
    error_type: Optional[str] = None
    started_at: str
    duration_ms: Optional[int] = None
    video_id: Optional[str] = None
    episode_id: Optional[str] = None


class LogEntry(BaseModel):
    ts: str
    level: str
    message: str
    error_type: Optional[str] = None


class Metric(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None


class Artifact(BaseModel):
    label: str
    uri: str


class JobDetail(JobRow):
    logs: List[LogEntry]
    metrics: Optional[List[Metric]] = None
    artifacts: Optional[List[Artifact]] = None


class VersionRow(BaseModel):
    dataset_name: str
    version: str
    status: VersionStatus
    created_at: str
    manifest_uri: str
    parent_version: Optional[str] = None
    total_episodes: int
    high_quality_ratio: float


class WeeklyQuality(BaseModel):
    week: str
    episodes: int
    high_quality: int
    conf_p50: float
    conf_p90: float
    jitter_p90: float
    interpolated_ratio_p90: float


class CostPoint(BaseModel):
    date: str
    storage_gb: float
    est_cost_usd: float


class JobsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    jobs: List[JobRow]


# ===== Data Generation (Mock) =====

def generate_throughput(hours: int = 24) -> List[ThroughputPoint]:
    """처리량 시계열 데이터 생성"""
    now = datetime.utcnow()
    points = []
    
    for i in range(hours):
        ts = (now - timedelta(hours=hours - 1 - i)).strftime("%Y-%m-%dT%H:00")
        jobs = int(50 + 20 * math.sin(i / 3) + random.random() * 15)
        errors = max(0, int(jobs * (0.01 + 0.02 * random.random())))
        points.append(ThroughputPoint(ts=ts, jobs=jobs, errors=errors))
    
    return points


def generate_jobs(count: int = 60) -> List[JobRow]:
    """작업 목록 생성"""
    stages = list(StageName)
    statuses = [JobStatus.SUCCESS] * 3 + [JobStatus.SKIP, JobStatus.FAIL, JobStatus.RUNNING]
    error_types = ["NETWORK_TRANSIENT", "RATE_LIMIT", "INTEGRITY_FAIL", "QUALITY_SKIP", "AUTHORIZATION"]
    
    jobs = []
    now = datetime.utcnow()
    
    for i in range(count):
        stage = stages[i % len(stages)]
        status = statuses[(i * 7) % len(statuses)]
        started = (now - timedelta(minutes=i * 6)).isoformat()
        
        job = JobRow(
            job_key=f"{stage.value}:youtube:vid_{1000 + i}:v4.0.0",
            run_id=f"run_{i // 10}",
            stage=stage,
            status=status,
            error_type=error_types[i % len(error_types)] if status == JobStatus.FAIL else None,
            started_at=started,
            duration_ms=int(1000 + random.random() * 250000) if status != JobStatus.RUNNING else None,
            video_id=f"vid_{1000 + i}",
            episode_id=f"vid_{1000 + i}_ep{str((i % 7) + 1).zfill(3)}" if stage == StageName.TRANSFORM else None,
        )
        jobs.append(job)
    
    return jobs


# ===== API Endpoints =====

@app.get("/api/overview", response_model=OverviewResponse)
async def get_overview(range: str = Query("24h", description="Time range (e.g., 24h, 7d)")):
    """
    파이프라인 개요 및 KPI
    """
    throughput = generate_throughput(24)
    total_jobs = sum(p.jobs for p in throughput)
    total_errors = sum(p.errors for p in throughput)
    error_rate = (total_errors / total_jobs * 100) if total_jobs > 0 else 0
    
    return OverviewResponse(
        range=range,
        kpi=KPI(
            total_videos=518,
            downloaded_videos=492,
            total_episodes=12640,
            high_quality_episodes=6721,
            storage_gb=38.4,
            monthly_cost_usd=9.83,
        ),
        health=Health(
            error_rate_pct=round(error_rate, 2),
            p95_end_to_end_ms=240000,
            queue_backlog=63,
            last_alert="upload: INTEGRITY_FAIL spike" if error_rate >= 1 else None,
        ),
        throughput=throughput,
    )


@app.get("/api/stages", response_model=List[StageStatus])
async def get_stages(range: str = Query("24h")):
    """
    스테이지별 상태
    """
    return [
        StageStatus(stage=StageName.DISCOVER, success=820, fail=7, skip=0, p95_ms=1800, inflight=2, queue_depth=4),
        StageStatus(stage=StageName.DOWNLOAD, success=492, fail=18, skip=8, p95_ms=220000, inflight=5, queue_depth=22),
        StageStatus(stage=StageName.EXTRACT, success=470, fail=9, skip=13, p95_ms=310000, inflight=4, queue_depth=15),
        StageStatus(stage=StageName.TRANSFORM, success=460, fail=5, skip=22, p95_ms=90000, inflight=3, queue_depth=9),
        StageStatus(stage=StageName.UPLOAD, success=445, fail=11, skip=0, p95_ms=120000, inflight=6, queue_depth=17),
        StageStatus(stage=StageName.FINALIZE, success=2, fail=0, skip=0, p95_ms=8000, inflight=0, queue_depth=0),
    ]


@app.get("/api/jobs", response_model=JobsResponse)
async def get_jobs(
    range: str = Query("24h"),
    stage: Optional[StageName] = None,
    status: Optional[JobStatus] = None,
    query: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    작업 목록 조회 (필터링 지원)
    """
    all_jobs = generate_jobs(60)
    
    # 필터링
    filtered = all_jobs
    if stage:
        filtered = [j for j in filtered if j.stage == stage]
    if status:
        filtered = [j for j in filtered if j.status == status]
    if query:
        q = query.lower()
        filtered = [j for j in filtered if q in j.job_key.lower() or (j.video_id and q in j.video_id.lower())]
    
    # 페이지네이션
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]
    
    return JobsResponse(
        total=total,
        page=page,
        page_size=page_size,
        jobs=paginated,
    )


@app.get("/api/job/{job_key}", response_model=JobDetail)
async def get_job_detail(job_key: str):
    """
    작업 상세 정보
    """
    all_jobs = generate_jobs(60)
    job = next((j for j in all_jobs if j.job_key == job_key), None)
    
    if not job:
        # 기본 작업 반환
        job = all_jobs[0]
    
    now = datetime.utcnow()
    
    logs = [
        LogEntry(ts=(now - timedelta(seconds=20)).isoformat(), level="INFO", message="job started"),
        LogEntry(ts=(now - timedelta(seconds=12)).isoformat(), level="INFO", message="downloaded candidate metadata"),
    ]
    
    if job.status == JobStatus.FAIL:
        logs.append(LogEntry(
            ts=(now - timedelta(seconds=6)).isoformat(),
            level="ERROR",
            message="upload failed: checksum mismatch",
            error_type=job.error_type,
        ))
    else:
        logs.append(LogEntry(
            ts=(now - timedelta(seconds=6)).isoformat(),
            level="INFO",
            message="job completed",
        ))
    
    return JobDetail(
        **job.model_dump(),
        logs=logs,
        metrics=[
            Metric(name="duration", value=job.duration_ms or 0, unit="ms"),
            Metric(name="bytes", value=int(3e6 + random.random() * 30e6), unit="B"),
            Metric(name="frames", value=int(2000 + random.random() * 9000)),
        ],
        artifacts=[
            Artifact(label="episode npz", uri="s3://bucket/dataset/versions/v1.2.0/episodes/..."),
            Artifact(label="manifest", uri="s3://bucket/dataset/versions/v1.2.0/manifests/manifest.json"),
        ],
    )


@app.get("/api/versions", response_model=List[VersionRow])
async def get_versions():
    """
    데이터셋 버전 목록
    """
    return [
        VersionRow(
            dataset_name="physicalai-motion",
            version="1.2.0",
            status=VersionStatus.RELEASED,
            created_at="2026-02-04T10:12:00Z",
            manifest_uri="s3://bucket/physicalai-motion/versions/v1.2.0/manifests/manifest.json",
            parent_version="1.1.0",
            total_episodes=12640,
            high_quality_ratio=0.53,
        ),
        VersionRow(
            dataset_name="physicalai-motion",
            version="1.2.1",
            status=VersionStatus.DRAFT,
            created_at="2026-02-05T03:20:00Z",
            manifest_uri="s3://bucket/physicalai-motion/versions/v1.2.1/manifests/manifest.json",
            parent_version="1.2.0",
            total_episodes=3100,
            high_quality_ratio=0.55,
        ),
        VersionRow(
            dataset_name="physicalai-motion",
            version="1.1.0",
            status=VersionStatus.DEPRECATED,
            created_at="2026-01-20T08:00:00Z",
            manifest_uri="s3://bucket/physicalai-motion/versions/v1.1.0/manifests/manifest.json",
            parent_version="1.0.0",
            total_episodes=8420,
            high_quality_ratio=0.48,
        ),
    ]


@app.get("/api/quality/weekly", response_model=List[WeeklyQuality])
async def get_weekly_quality(weeks: int = Query(8, ge=1, le=52)):
    """
    주간 품질 리포트
    """
    data = []
    base_episodes = 800
    
    for i in range(weeks):
        week_num = 52 - weeks + i + 1 if weeks <= 52 else i + 1
        week = f"2026-W{str(week_num).zfill(2)}"
        episodes = base_episodes + i * 400 + int(random.random() * 200)
        high_quality = int(episodes * (0.48 + i * 0.01 + random.random() * 0.05))
        
        data.append(WeeklyQuality(
            week=week,
            episodes=episodes,
            high_quality=high_quality,
            conf_p50=0.74 + i * 0.005 + random.random() * 0.02,
            conf_p90=0.89 + i * 0.003 + random.random() * 0.01,
            jitter_p90=0.048 + random.random() * 0.01 - 0.005,
            interpolated_ratio_p90=0.18 + random.random() * 0.06,
        ))
    
    return data


@app.get("/api/cost", response_model=List[CostPoint])
async def get_cost(range: str = Query("30d")):
    """
    스토리지 비용 추적
    """
    days = int(range.replace("d", "")) if "d" in range else 30
    days = min(days, 90)
    
    data = []
    base_storage = 22.0
    now = datetime.utcnow()
    
    for i in range(days):
        date = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        storage_gb = round(base_storage + i * 0.6 + random.random() * 1.0, 1)
        est_cost_usd = round(storage_gb * 0.023 + 1.2 + random.random() * 0.3, 2)
        
        data.append(CostPoint(
            date=date,
            storage_gb=storage_gb,
            est_cost_usd=est_cost_usd,
        ))
    
    return data


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ===== Real Data Integration =====

def get_real_overview_from_db(db_session, range_hours: int = 24):
    """
    실제 DB에서 Overview 데이터 조회
    
    TODO: DB 연동 시 구현
    """
    try:
        from models.database import Video, Episode, ProcessingJob
        from sqlalchemy import func
        
        # 비디오 통계
        total_videos = db_session.query(func.count(Video.id)).scalar() or 0
        downloaded_videos = db_session.query(func.count(Video.id)).filter(
            Video.status == 'downloaded'
        ).scalar() or 0
        
        # 에피소드 통계
        total_episodes = db_session.query(func.count(Episode.id)).scalar() or 0
        high_quality = db_session.query(func.count(Episode.id)).filter(
            Episode.quality_score >= 0.7
        ).scalar() or 0
        
        # 작업 통계
        time_threshold = datetime.utcnow() - timedelta(hours=range_hours)
        jobs = db_session.query(ProcessingJob).filter(
            ProcessingJob.created_at >= time_threshold
        ).all()
        
        total_jobs = len(jobs)
        failed_jobs = len([j for j in jobs if j.status == 'failed'])
        error_rate = (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        return {
            "kpi": {
                "total_videos": total_videos,
                "downloaded_videos": downloaded_videos,
                "total_episodes": total_episodes,
                "high_quality_episodes": high_quality,
            },
            "health": {
                "error_rate_pct": round(error_rate, 2),
            }
        }
    except Exception as e:
        return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
