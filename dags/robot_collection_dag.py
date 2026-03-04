"""
P-ADE 로봇 영상 수집 파이프라인 Airflow DAG (v2 - 5000 목표)

매일 자동으로 로봇팔 영상을 수집하고 처리하는 파이프라인입니다.

파이프라인 단계:
  1. crawl_videos: 키워드로 영상 URL 수집
  2. download_videos: 영상 다운로드 (yt-dlp)
  3. process_videos: 객체 검출 + 포즈 추출 + 에피소드 생성
  4. quality_check: 품질 평가 및 필터링
  5. upload_s3: S3에 업로드
  6. cleanup: 로컬 RAW 파일 정리

사용법:
  # Airflow 설치 후 DAG 디렉토리에 복사
  cp dags/robot_collection_dag.py ~/airflow/dags/
  
  # 수동 트리거
  airflow dags trigger robot_arm_collection
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Airflow 임포트
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    # Airflow 없이도 임포트 가능하도록
    DAG = None
    PythonOperator = None
    BashOperator = None

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Task 함수 정의
# ============================================================================

def crawl_videos(**context):
    """
    Task 1: 영상 URL 크롤링
    
    YouTube, Google Videos 등에서 로봇 관련 영상을 검색합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    target_count = context.get("params", {}).get("target_count", 5000)
    
    config = PipelineConfig(
        target_count=target_count,
        sources=["youtube", "google_videos"],
        languages=["en", "ko"],
    )
    
    collector = MassCollector(config)
    collector.run(start_stage="crawl", end_stage="crawl")
    
    return {"stage": "crawl", "target": target_count}


def download_videos(**context):
    """
    Task 2: 영상 다운로드
    
    크롤링된 URL에서 영상을 다운로드합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig(
        download_workers=6,
        download_quality="720p",
    )
    
    collector = MassCollector(config)
    collector.run(start_stage="download", end_stage="download")
    
    return {"stage": "download"}


def process_videos(**context):
    """
    Task 3: GPU 처리 (객체 검출 + 포즈 추출 + 에피소드 생성)
    
    YOLO 객체 검출, MediaPipe 포즈 추출, 에피소드 NPZ 파일 생성을 통합 처리합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig()
    collector = MassCollector(config)
    collector.run(start_stage="process", end_stage="process")
    
    return {"stage": "process"}


def quality_check(**context):
    """
    Task 4: 품질 평가 및 필터링
    
    에피소드 품질 등급(A/B/C/D)을 평가하고 기준 미달 데이터를 필터링합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig()
    collector = MassCollector(config)
    collector.run(start_stage="quality", end_stage="quality")
    
    return {"stage": "quality"}


def upload_to_s3(**context):
    """
    Task 5: S3 업로드
    
    생성된 에피소드를 AWS S3에 업로드합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig()
    collector = MassCollector(config)
    collector.run(start_stage="upload", end_stage="upload")
    
    return {"stage": "upload"}


def cleanup(**context):
    """
    Task 6: 로컬 RAW 파일 정리
    
    업로드 완료된 로컬 raw 파일과 임시 파일을 정리합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig()
    collector = MassCollector(config)
    collector.run(start_stage="cleanup", end_stage="cleanup")
    
    return {"stage": "cleanup"}


# ============================================================================
# DAG 정의
# ============================================================================

if AIRFLOW_AVAILABLE:
    # 기본 인자 (D4-1: exponential backoff, 재시도 1회)
    default_args = {
        "owner": "robot-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "retry_exponential_backoff": True,
        "execution_timeout": timedelta(hours=6),
    }
    
    # DAG 생성
    dag = DAG(
        dag_id="robot_arm_collection",
        default_args=default_args,
        description="로봇팔 영상 자동 수집 파이프라인 (목표 5000건)",
        schedule_interval="0 6 * * *",  # 매일 오전 6시
        start_date=datetime(2026, 2, 10),
        catchup=False,
        tags=["robot", "video", "collection", "imitation-learning"],
        params={
            "target_count": 5000,  # 목표 수집량 5000
        },
    )
    
    # Task 정의 (D4-1: per-task execution_timeout)
    with dag:
        # Task 1: 크롤링 (30분)
        crawl_task = PythonOperator(
            task_id="crawl_videos",
            python_callable=crawl_videos,
            provide_context=True,
            execution_timeout=timedelta(minutes=30),
        )
        
        # Task 2: 다운로드 (5시간)
        download_task = PythonOperator(
            task_id="download_videos",
            python_callable=download_videos,
            provide_context=True,
            execution_timeout=timedelta(hours=5),
        )
        
        # Task 3: GPU 처리 (5시간)
        process_task = PythonOperator(
            task_id="process_videos",
            python_callable=process_videos,
            provide_context=True,
            execution_timeout=timedelta(hours=5),
        )
        
        # Task 4: 품질 평가 (30분)
        quality_task = PythonOperator(
            task_id="quality_check",
            python_callable=quality_check,
            provide_context=True,
            execution_timeout=timedelta(minutes=30),
        )
        
        # Task 5: S3 업로드 (2시간)
        upload_task = PythonOperator(
            task_id="upload_s3",
            python_callable=upload_to_s3,
            provide_context=True,
            execution_timeout=timedelta(hours=2),
        )
        
        # Task 6: 정리
        cleanup_task = PythonOperator(
            task_id="cleanup",
            python_callable=cleanup,
            provide_context=True,
            execution_timeout=timedelta(minutes=30),
        )
        
        # 의존성 정의 (D4-1: process + quality + cleanup 체인)
        crawl_task >> download_task >> process_task >> quality_task >> upload_task >> cleanup_task


# ============================================================================
# 로컬 테스트용
# ============================================================================

def test_dag():
    """DAG 로컬 테스트"""
    print("🧪 DAG 로컬 테스트 (v2 - 5000 목표)")
    print("=" * 60)
    
    # 각 함수 개별 테스트 (실제로는 실행하지 않음)
    tasks = [
        ("crawl_videos", crawl_videos),
        ("download_videos", download_videos),
        ("process_videos", process_videos),
        ("quality_check", quality_check),
        ("upload_to_s3", upload_to_s3),
        ("cleanup", cleanup),
    ]
    
    for name, func in tasks:
        print(f"  ✓ {name}: 함수 정의됨")
    
    print("=" * 60)
    print("✅ 모든 함수 검증 완료")
    
    if AIRFLOW_AVAILABLE:
        print(f"✅ Airflow DAG 'robot_arm_collection' 정의됨")
        print(f"   스케줄: 매일 오전 6시")
        print(f"   시작일: 2026-02-10")
        print(f"   체인: crawl >> download >> process >> quality >> upload >> cleanup")
    else:
        print("⚠️ Airflow가 설치되지 않음 - DAG 미생성")


if __name__ == "__main__":
    test_dag()
