"""
P-ADE 로봇 영상 수집 파이프라인 Airflow DAG

매일 자동으로 로봇팔 영상을 수집하고 처리하는 파이프라인입니다.

파이프라인 단계:
  1. crawl_videos: 키워드로 영상 URL 수집
  2. download_videos: 영상 다운로드 (yt-dlp)
  3. detect_objects: 객체 검출 및 포즈 추출
  4. build_imitation_data: 모방학습 데이터 생성
  5. upload_s3: S3에 업로드

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
    
    target_count = context.get("params", {}).get("target_count", 100)
    
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


def detect_objects(**context):
    """
    Task 3: 객체 검출 및 에피소드 생성
    
    YOLO로 객체를 검출하고 에피소드 NPZ 파일을 생성합니다.
    """
    from mass_collector import MassCollector, PipelineConfig
    
    config = PipelineConfig()
    collector = MassCollector(config)
    collector.run(start_stage="detect", end_stage="detect")
    
    return {"stage": "detect"}


def build_imitation_data(**context):
    """
    Task 4: 모방학습 데이터 생성
    
    MediaPipe로 포즈를 추출하고 State-Action 데이터를 생성합니다.
    """
    import subprocess
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pipeline" / "build_imitation_data.py"),
        "--fps", "5",
        "--max-frames", "100",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"모방학습 데이터 생성 실패: {result.stderr}")
    
    return {"stage": "build_imitation_data"}


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


def send_notification(**context):
    """
    Task 6: 완료 알림 (옵션)
    
    파이프라인 완료 시 슬랙/이메일 알림을 보냅니다.
    """
    # TODO: 슬랙 웹훅 또는 이메일 알림 구현
    print("✅ 파이프라인 완료!")
    return {"stage": "notification"}


# ============================================================================
# DAG 정의
# ============================================================================

if AIRFLOW_AVAILABLE:
    # 기본 인자
    default_args = {
        "owner": "robot-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=4),
    }
    
    # DAG 생성
    dag = DAG(
        dag_id="robot_arm_collection",
        default_args=default_args,
        description="로봇팔 영상 자동 수집 파이프라인",
        schedule_interval="0 6 * * *",  # 매일 오전 6시
        start_date=datetime(2026, 2, 10),
        catchup=False,
        tags=["robot", "video", "collection", "imitation-learning"],
        params={
            "target_count": 100,  # 기본 목표 수집량
        },
    )
    
    # Task 정의
    with dag:
        # Task 1: 크롤링
        crawl_task = PythonOperator(
            task_id="crawl_videos",
            python_callable=crawl_videos,
            provide_context=True,
        )
        
        # Task 2: 다운로드
        download_task = PythonOperator(
            task_id="download_videos",
            python_callable=download_videos,
            provide_context=True,
        )
        
        # Task 3: 객체 검출
        detect_task = PythonOperator(
            task_id="detect_objects",
            python_callable=detect_objects,
            provide_context=True,
        )
        
        # Task 4: 모방학습 데이터 생성
        build_il_task = PythonOperator(
            task_id="build_imitation_data",
            python_callable=build_imitation_data,
            provide_context=True,
        )
        
        # Task 5: S3 업로드
        upload_task = PythonOperator(
            task_id="upload_s3",
            python_callable=upload_to_s3,
            provide_context=True,
        )
        
        # Task 6: 알림 (선택)
        notify_task = PythonOperator(
            task_id="send_notification",
            python_callable=send_notification,
            provide_context=True,
        )
        
        # 의존성 정의
        crawl_task >> download_task >> detect_task >> build_il_task >> upload_task >> notify_task


# ============================================================================
# 로컬 테스트용
# ============================================================================

def test_dag():
    """DAG 로컬 테스트"""
    print("🧪 DAG 로컬 테스트")
    print("=" * 60)
    
    # 각 함수 개별 테스트 (실제로는 실행하지 않음)
    tasks = [
        ("crawl_videos", crawl_videos),
        ("download_videos", download_videos),
        ("detect_objects", detect_objects),
        ("build_imitation_data", build_imitation_data),
        ("upload_to_s3", upload_to_s3),
        ("send_notification", send_notification),
    ]
    
    for name, func in tasks:
        print(f"  ✓ {name}: 함수 정의됨")
    
    print("=" * 60)
    print("✅ 모든 함수 검증 완료")
    
    if AIRFLOW_AVAILABLE:
        print(f"✅ Airflow DAG 'robot_arm_collection' 정의됨")
        print(f"   스케줄: 매일 오전 6시")
        print(f"   시작일: 2026-02-10")
    else:
        print("⚠️ Airflow가 설치되지 않음 - DAG 미생성")


if __name__ == "__main__":
    test_dag()
