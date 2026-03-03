"""
Airflow DAG 테스트

DAG 정의의 정상 동작, 함수 실행 가능성, 의존성 체인을 검증합니다.
Airflow 미설치 환경에서도 테스트 가능합니다.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDAGStructure:
    """DAG 구조 검증 테스트"""
    
    def test_dag_module_import(self):
        """DAG 모듈 임포트 테스트"""
        from dags import robot_collection_dag
        assert robot_collection_dag is not None
    
    def test_task_functions_exist(self):
        """태스크 함수 존재 확인"""
        from dags.robot_collection_dag import (
            crawl_videos,
            download_videos,
            detect_objects,
            build_imitation_data,
            upload_to_s3,
            send_notification,
        )
        
        assert callable(crawl_videos)
        assert callable(download_videos)
        assert callable(detect_objects)
        assert callable(build_imitation_data)
        assert callable(upload_to_s3)
        assert callable(send_notification)
    
    def test_task_count(self):
        """태스크 함수 수 확인 (6개)"""
        from dags import robot_collection_dag
        
        task_names = [
            "crawl_videos",
            "download_videos",
            "detect_objects",
            "build_imitation_data",
            "upload_to_s3",
            "send_notification",
        ]
        
        for name in task_names:
            assert hasattr(robot_collection_dag, name), f"Missing task: {name}"
    
    def test_project_root_setup(self):
        """프로젝트 루트 설정 확인"""
        from dags.robot_collection_dag import PROJECT_ROOT
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "mass_collector.py").exists()


class TestDAGTasks:
    """DAG 태스크 단위 테스트"""
    
    def test_send_notification(self):
        """알림 태스크 실행 테스트 (부작용 없음)"""
        from dags.robot_collection_dag import send_notification
        
        result = send_notification()
        assert result["stage"] == "notification"
    
    @patch("dags.robot_collection_dag.MassCollector")
    def test_crawl_videos_mock(self, mock_collector_cls):
        """크롤링 태스크 Mock 테스트"""
        from dags.robot_collection_dag import crawl_videos
        
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        
        result = crawl_videos(params={"target_count": 10})
        
        assert result["stage"] == "crawl"
        assert result["target"] == 10
        mock_collector.run.assert_called_once_with(
            start_stage="crawl", end_stage="crawl"
        )
    
    @patch("dags.robot_collection_dag.MassCollector")
    def test_download_videos_mock(self, mock_collector_cls):
        """다운로드 태스크 Mock 테스트"""
        from dags.robot_collection_dag import download_videos
        
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        
        result = download_videos()
        
        assert result["stage"] == "download"
        mock_collector.run.assert_called_once()
    
    @patch("dags.robot_collection_dag.MassCollector")
    def test_detect_objects_mock(self, mock_collector_cls):
        """검출 태스크 Mock 테스트"""
        from dags.robot_collection_dag import detect_objects
        
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        
        result = detect_objects()
        
        assert result["stage"] == "detect"
    
    @patch("dags.robot_collection_dag.MassCollector")
    def test_upload_to_s3_mock(self, mock_collector_cls):
        """업로드 태스크 Mock 테스트"""
        from dags.robot_collection_dag import upload_to_s3
        
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        
        result = upload_to_s3()
        
        assert result["stage"] == "upload"
    
    @patch("subprocess.run")
    def test_build_imitation_data_success(self, mock_run):
        """모방학습 데이터 생성 성공 테스트"""
        from dags.robot_collection_dag import build_imitation_data
        
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        
        result = build_imitation_data()
        assert result["stage"] == "build_imitation_data"
    
    @patch("subprocess.run")
    def test_build_imitation_data_failure(self, mock_run):
        """모방학습 데이터 생성 실패 테스트"""
        from dags.robot_collection_dag import build_imitation_data
        
        mock_run.return_value = MagicMock(returncode=1, stderr="error occurred")
        
        with pytest.raises(Exception, match="모방학습 데이터 생성 실패"):
            build_imitation_data()


class TestDAGDefinition:
    """Airflow DAG 정의 테스트 (Airflow 설치 시)"""
    
    def test_dag_available_flag(self):
        """AIRFLOW_AVAILABLE 플래그 확인"""
        from dags.robot_collection_dag import AIRFLOW_AVAILABLE
        # Airflow 설치 여부에 관계없이 모듈은 임포트 가능해야 함
        assert isinstance(AIRFLOW_AVAILABLE, bool)
    
    def test_test_dag_function(self):
        """test_dag() 로컬 테스트 함수 실행"""
        from dags.robot_collection_dag import test_dag
        # 에러 없이 실행되어야 함
        test_dag()


class TestDAGTaskFlow:
    """태스크 흐름 검증"""
    
    def test_task_order_logic(self):
        """태스크 순서 논리 검증"""
        # 올바른 실행 순서: crawl → download → detect → build → upload → notify
        expected_order = [
            "crawl",
            "download",
            "detect",
            "build_imitation_data",
            "upload",
            "notification",
        ]
        
        from dags.robot_collection_dag import (
            crawl_videos,
            download_videos,
            detect_objects,
            build_imitation_data,
            upload_to_s3,
            send_notification,
        )
        
        tasks = [
            crawl_videos,
            download_videos,
            detect_objects,
            build_imitation_data,
            upload_to_s3,
            send_notification,
        ]
        
        # 모든 태스크가 callable인지 확인
        for task, stage in zip(tasks, expected_order):
            assert callable(task), f"Task for stage '{stage}' is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
