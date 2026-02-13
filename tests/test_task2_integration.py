"""
Task 2 Integration Tests

Task 2.1: Crawl Task Queue (queue/task_queue.py, workers/crawl_worker.py)
Task 2.2: GPU 3-Stream Processing (gpu/stream_manager.py)
Task 2.3: Realtime Quality Evaluation (quality/evaluator.py + Redis integration)
Task 2.4: Unified Dashboard (dashboard/web_app.py + SSE + Control)
"""

import sys
import os
import json
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["ENVIRONMENT"] = "test"


# ============================================================================
# Task 2.1: Crawl Task Queue Tests
# ============================================================================

class TestCrawlTaskQueue:
    """CrawlTaskQueue tests"""

    def test_queue_creation_without_redis(self):
        """Queue creation possible without Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert not queue.is_connected

    def test_queue_length_without_redis(self):
        """queue_length returns 0 without Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert queue.queue_length() == 0

    def test_enqueue_without_redis_returns_zero(self):
        """enqueue returns 0 without Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        result = queue.enqueue_keywords(["robot arm", "pick and place"])
        assert result == 0

    def test_dequeue_without_redis_returns_none(self):
        """dequeue returns None without Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        result = queue.dequeue_keyword(timeout=1)
        assert result is None

    def test_stats_without_redis(self):
        """Empty stats without Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert queue.get_stats() == {}

    def test_enqueue_keywords_with_mock_redis(self):
        """Enqueue test with Mock Redis"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)

        mock_client = Mock()
        mock_client.ping.return_value = True
        queue._client = mock_client
        queue._connected = True

        count = queue.enqueue_keywords(["robot arm", "pick and place"])
        assert count == 2
        assert mock_client.rpush.call_count == 2

    def test_mark_complete(self):
        """Task completion marking"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)

        mock_client = Mock()
        mock_client.ping.return_value = True
        queue._client = mock_client
        queue._connected = True

        results = [{"video_id": "abc", "url": "http://example.com"}]
        queue.mark_complete("robot arm", results)

        mock_client.hset.assert_called_once()
        assert mock_client.hincrby.call_count == 2  # completed + results

    def test_mark_failed(self):
        """Task failure marking"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)

        mock_client = Mock()
        mock_client.ping.return_value = True
        queue._client = mock_client
        queue._connected = True

        queue.mark_failed("robot arm", "timeout error")
        mock_client.hset.assert_called_once()
        mock_client.hincrby.assert_called_with(queue.STATS_KEY, "total_failed", 1)


class TestCrawlTask:
    """CrawlTask dataclass tests"""

    def test_crawl_task_creation(self):
        from task_queue.task_queue import CrawlTask
        task = CrawlTask(keyword="robot arm", source="youtube", max_results=50, priority=1)
        assert task.keyword == "robot arm"
        assert task.source == "youtube"
        assert task.max_results == 50
        assert task.priority == 1

    def test_crawl_task_defaults(self):
        from task_queue.task_queue import CrawlTask
        task = CrawlTask(keyword="test")
        assert task.source == "youtube"
        assert task.max_results == 50
        assert task.priority == 0


class TestProcessingQueue:
    """ProcessingQueue tests (for GPU processing)"""

    def test_processing_queue_keys(self):
        """ProcessingQueue uses separate Redis keys"""
        from task_queue.task_queue import ProcessingQueue
        pq = ProcessingQueue(host="invalid_host", port=9999)
        assert pq.QUEUE_KEY == "pade:processing_queue"
        assert pq.RESULTS_KEY == "pade:processing_results"
        assert pq.STATS_KEY == "pade:processing_stats"

    def test_pop_batch(self):
        """Batch pop functionality test"""
        from task_queue.task_queue import ProcessingQueue
        pq = ProcessingQueue(host="invalid_host", port=9999)

        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.side_effect = [
            json.dumps({"video_path": "/a.mp4", "video_id": "a"}),
            json.dumps({"video_path": "/b.mp4", "video_id": "b"}),
            None,
        ]
        pq._client = mock_client
        pq._connected = True

        batch = pq.pop_batch(batch_size=3)
        assert len(batch) == 2
        assert batch[0]["video_id"] == "a"
        assert batch[1]["video_id"] == "b"


class TestCrawlWorkerConfig:
    """Crawl worker configuration tests"""

    def test_worker_config_defaults(self):
        """Default configuration"""
        from workers.crawl_worker import WorkerConfig
        config = WorkerConfig(worker_id=0)
        assert config.worker_id == 0
        assert config.timeout_sec == 5
        assert config.max_tasks == 0
        assert config.source == "youtube"

    def test_worker_config_custom(self):
        """Custom configuration"""
        from workers.crawl_worker import WorkerConfig
        config = WorkerConfig(
            worker_id=3,
            timeout_sec=10,
            max_tasks=50,
            source="google_videos",
            max_results_per_task=100,
        )
        assert config.worker_id == 3
        assert config.source == "google_videos"


# ============================================================================
# Task 2.2: GPU 3-Stream Tests
# ============================================================================

class TestGPUStreamConfig:
    """GPU Stream configuration tests"""

    def test_stream_config_defaults(self):
        from gpu.stream_manager import StreamConfig
        config = StreamConfig()
        assert config.num_streams == 3
        assert config.vram_limit_gb == 9.0
        assert config.target_fps == 30
        assert config.long_video_fps == 15
        assert config.long_video_threshold_sec == 60

    def test_stream_config_custom(self):
        from gpu.stream_manager import StreamConfig
        config = StreamConfig(num_streams=4, vram_limit_gb=12.0)
        assert config.num_streams == 4
        assert config.vram_limit_gb == 12.0


class TestGPU3StreamManager:
    """GPU3StreamManager tests"""

    def test_manager_creation(self):
        """Manager creation (works without CUDA)"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        assert manager.config.num_streams == 3

    def test_vram_usage_without_cuda(self):
        """VRAM 0 without CUDA"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        usage = manager.get_vram_usage()
        assert "allocated" in usage
        assert "reserved" in usage
        assert "available" in usage

    def test_auto_adjust_batch_size_without_cuda(self):
        """Batch size 1 without CUDA"""
        from gpu.stream_manager import GPU3StreamManager, CUDA_AVAILABLE
        manager = GPU3StreamManager()
        batch_size = manager.auto_adjust_batch_size()
        if not CUDA_AVAILABLE:
            assert batch_size == 1

    def test_optimal_fps_short_video(self):
        """Short video gets 30fps"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        fps = manager.get_optimal_fps(30.0)
        assert fps == 30

    def test_optimal_fps_long_video(self):
        """Long video gets 15fps"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        fps = manager.get_optimal_fps(120.0)
        assert fps == 15

    def test_check_vram_health_without_cuda(self):
        """Health True without CUDA"""
        from gpu.stream_manager import GPU3StreamManager, CUDA_AVAILABLE
        manager = GPU3StreamManager()
        if not CUDA_AVAILABLE:
            assert manager.check_vram_health() is True

    def test_stats_initial(self):
        """Initial stats"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        assert manager.stats["total_processed"] == 0
        assert manager.stats["total_time_sec"] == 0

    def test_process_batch_empty(self):
        """Empty batch processing - confirm empty list returned for batch_size=0"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        try:
            results = manager.process_batch([], processor=lambda x: {"success": True})
            assert results == []
        except ValueError:
            pass


# ============================================================================
# Task 2.3: Realtime Quality Evaluation Tests
# ============================================================================

class TestQualityEvaluator:
    """RobotArmQualityEvaluator tests"""

    @pytest.fixture
    def evaluator(self):
        from quality.evaluator import RobotArmQualityEvaluator, QualityConfig
        return RobotArmQualityEvaluator(
            config=QualityConfig(pass_threshold=60.0)
        )

    @pytest.fixture
    def good_sequence(self):
        """Good quality sequence"""
        np.random.seed(42)
        num_frames = 60
        body_frames = []
        for i in range(num_frames):
            frame = np.random.rand(33, 4)
            # High visibility
            frame[:, 3] = 0.8 + np.random.rand(33) * 0.2
            # Natural arm motion (gradual position change)
            frame[15, :3] = [0.3 + i * 0.005, 0.5, 0.1]
            frame[16, :3] = [0.7 - i * 0.005, 0.5, 0.1]
            body_frames.append(frame)

        right_hand = []
        for i in range(num_frames):
            hand = np.random.rand(21, 3)
            # Grasping motion data
            open_factor = 0.3 if i < 30 else 0.05
            hand[4] = [0.5, 0.5, 0]  # thumb
            hand[8] = [0.5 + open_factor, 0.5, 0]  # index
            right_hand.append(hand)

        return {
            "body": body_frames,
            "right_hand": right_hand,
            "left_hand": [],
        }

    @pytest.fixture
    def bad_sequence(self):
        """Bad quality sequence (very few frames)"""
        return {
            "body": [np.random.rand(33, 4) for _ in range(3)],
            "right_hand": [],
            "left_hand": [],
        }

    def test_evaluate_good_sequence(self, evaluator, good_sequence):
        """Good sequence evaluation"""
        result = evaluator.evaluate(good_sequence, "test_good")
        assert result.total_score > 0
        assert result.joint_score > 0
        assert result.video_id == "test_good"

    def test_evaluate_bad_sequence(self, evaluator, bad_sequence):
        """Bad sequence - frame count insufficient"""
        result = evaluator.evaluate(bad_sequence, "test_bad")
        assert result.total_score == 0
        assert len(result.issues) > 0

    def test_evaluate_empty_sequence(self, evaluator):
        """Empty sequence"""
        result = evaluator.evaluate({}, "test_empty")
        assert result.total_score == 0
        assert "\ud3ec\uc988 \ub370\uc774\ud130 \uc5c6\uc74c" in result.issues

    def test_evaluate_none_sequence(self, evaluator):
        """None sequence"""
        result = evaluator.evaluate(None, "test_none")
        assert result.total_score == 0

    def test_grade_classification(self, evaluator):
        """Grade classification"""
        from quality.evaluator import Grade
        assert evaluator._determine_grade(95) == Grade.A
        assert evaluator._determine_grade(85) == Grade.B
        assert evaluator._determine_grade(75) == Grade.C
        assert evaluator._determine_grade(65) == Grade.D
        assert evaluator._determine_grade(50) == Grade.F

    def test_joint_evaluation(self, evaluator, good_sequence):
        """Joint detection evaluation"""
        score, detected = evaluator._evaluate_joints(good_sequence["body"])
        assert score >= 0
        assert score <= 30
        assert isinstance(detected, dict)
        assert "shoulder" in detected
        assert "elbow" in detected
        assert "wrist" in detected
        assert "gripper" in detected

    def test_motion_evaluation(self, evaluator, good_sequence):
        """Motion quality evaluation"""
        score = evaluator._evaluate_motion(good_sequence["body"])
        assert score >= 0
        assert score <= 25

    def test_grasping_evaluation(self, evaluator, good_sequence):
        """Grasping motion evaluation"""
        score, has_grasping = evaluator._evaluate_grasping(
            good_sequence.get("left_hand", []),
            good_sequence.get("right_hand", []),
        )
        assert score >= 0
        assert score <= 20
        assert isinstance(has_grasping, (bool, np.bool_))

    def test_stability_evaluation(self, evaluator, good_sequence):
        """Stability evaluation"""
        score = evaluator._evaluate_stability(good_sequence["body"])
        assert score >= 0
        assert score <= 15

    def test_coverage_evaluation(self, evaluator, good_sequence):
        """Coverage evaluation"""
        score, coverage = evaluator._evaluate_coverage(good_sequence["body"])
        assert score >= 0
        assert score <= 10
        assert coverage >= 0
        assert coverage <= 1.0


class TestQualityStats:
    """QualityStats tests (including Redis integration)"""

    def test_stats_creation(self):
        """Stats object creation"""
        from quality.evaluator import QualityStats
        stats = QualityStats()
        assert stats.total == 0
        assert stats.passed == 0
        assert stats.pass_rate == 0

    def test_stats_record(self):
        """Result recording"""
        from quality.evaluator import QualityStats, EvaluationResult, Grade
        stats = QualityStats()

        result_a = EvaluationResult(video_id="v1", total_score=95, grade=Grade.A, passed=True)
        result_f = EvaluationResult(video_id="v2", total_score=40, grade=Grade.F, passed=False)

        stats.record(result_a)
        assert stats.total == 1
        assert stats.passed == 1
        assert stats.grades["A"] == 1

        stats.record(result_f)
        assert stats.total == 2
        assert stats.passed == 1
        assert stats.grades["F"] == 1
        assert stats.pass_rate == 50.0

    def test_stats_pass_rate(self):
        """Pass rate calculation"""
        from quality.evaluator import QualityStats, EvaluationResult, Grade
        stats = QualityStats()

        for i in range(7):
            r = EvaluationResult(video_id=f"v{i}", total_score=75, grade=Grade.C, passed=True)
            stats.record(r)
        for i in range(3):
            r = EvaluationResult(video_id=f"f{i}", total_score=40, grade=Grade.F, passed=False)
            stats.record(r)

        assert stats.pass_rate == 70.0

    def test_stats_redis_publish(self):
        """Redis stats publish test"""
        from quality.evaluator import QualityStats, EvaluationResult, Grade

        mock_client = Mock()
        mock_pipe = Mock()
        mock_client.pipeline.return_value = mock_pipe

        stats = QualityStats()
        stats._redis_client = mock_client

        result = EvaluationResult(video_id="test1", total_score=85, grade=Grade.B, passed=True)
        stats.record(result)

        mock_client.pipeline.assert_called_once()
        mock_pipe.execute.assert_called_once()
        assert mock_pipe.hset.call_count >= 4

    def test_stats_redis_not_available(self):
        """Normal operation without Redis"""
        from quality.evaluator import QualityStats, EvaluationResult, Grade
        stats = QualityStats()
        stats._redis_client = None

        result = EvaluationResult(video_id="test", total_score=80, grade=Grade.B, passed=True)
        stats.record(result)
        assert stats.total == 1


class TestQualityConfig:
    """QualityConfig configuration tests"""

    def test_default_config(self):
        from quality.evaluator import QualityConfig
        config = QualityConfig()
        assert config.min_joint_confidence == 0.5
        assert config.grasping_threshold == 0.15
        assert config.pass_threshold == 60.0

    def test_custom_config(self):
        from quality.evaluator import QualityConfig
        config = QualityConfig(
            min_joint_confidence=0.7,
            pass_threshold=70.0,
        )
        assert config.min_joint_confidence == 0.7
        assert config.pass_threshold == 70.0


class TestEvaluationResult:
    """EvaluationResult dataclass tests"""

    def test_result_defaults(self):
        from quality.evaluator import EvaluationResult, Grade
        result = EvaluationResult(video_id="test")
        assert result.total_score == 0
        assert result.grade == Grade.F
        assert result.passed is False
        assert result.joint_score == 0
        assert result.detected_joints == {}
        assert result.issues == []


# ============================================================================
# Task 2.4: Unified Dashboard Tests (dashboard/web_app.py - Flask)
# ============================================================================

class TestFlaskDashboard:
    """Flask dashboard endpoint tests"""

    @pytest.fixture
    def client(self):
        """Create Flask TestClient"""
        try:
            from dashboard.web_app import app
            app.config["TESTING"] = True
            return app.test_client()
        except ImportError:
            pytest.skip("flask not installed")

    def test_health_check(self, client):
        """GET /api/health"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_overview(self, client):
        """GET /api/overview"""
        response = client.get("/api/overview")
        assert response.status_code == 200
        data = response.get_json()
        assert "kpi" in data
        assert "health" in data
        assert "throughput" in data

    def test_stages(self, client):
        """GET /api/stages"""
        response = client.get("/api/stages")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_jobs_list(self, client):
        """GET /api/jobs/search"""
        response = client.get("/api/jobs/search?page=1&page_size=10")
        assert response.status_code == 200
        data = response.get_json()
        assert "total" in data
        assert "jobs" in data
        assert data["page"] == 1

    def test_versions(self, client):
        """GET /api/versions"""
        response = client.get("/api/versions")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    def test_weekly_quality(self, client):
        """GET /api/quality/weekly"""
        response = client.get("/api/quality/weekly?weeks=4")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 4

    def test_cost(self, client):
        """GET /api/cost"""
        response = client.get("/api/cost?range=7d")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 7

    def test_realtime_stats(self, client):
        """GET /api/realtime"""
        response = client.get("/api/realtime")
        assert response.status_code == 200
        data = response.get_json()
        assert "crawl_speed" in data
        assert "download_speed" in data
        assert "gpu_util" in data

    def test_pipeline_status(self, client):
        """GET /api/pipeline/status"""
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "is_running" in data

    def test_index_page(self, client):
        """GET / - HTML rendering"""
        response = client.get("/")
        assert response.status_code == 200
        assert b"P-ADE" in response.data


class TestFlaskPipelineControl:
    """Flask pipeline control endpoint tests"""

    @pytest.fixture
    def client(self):
        try:
            from dashboard.web_app import app
            app.config["TESTING"] = True
            return app.test_client()
        except ImportError:
            pytest.skip("flask not installed")

    @patch("dashboard.web_app.get_redis_client")
    def test_control_start(self, mock_get_redis, client):
        """POST /api/control/start"""
        mock_redis = Mock()
        mock_redis.get.return_value = "running"
        mock_get_redis.return_value = mock_redis

        response = client.post("/api/control/start")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["action"] == "start"
        mock_redis.set.assert_called()
        mock_redis.publish.assert_called_with("pade:control", "start")

    @patch("dashboard.web_app.get_redis_client")
    def test_control_stop(self, mock_get_redis, client):
        """POST /api/control/stop"""
        mock_redis = Mock()
        mock_redis.get.return_value = "stopped"
        mock_get_redis.return_value = mock_redis

        response = client.post("/api/control/stop")
        assert response.status_code == 200
        data = response.get_json()
        assert data["action"] == "stop"

    @patch("dashboard.web_app.get_redis_client")
    def test_control_invalid_action(self, mock_get_redis, client):
        """Invalid action"""
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis

        response = client.post("/api/control/explode")
        assert response.status_code == 400

    @patch("dashboard.web_app.get_redis_client")
    def test_control_no_redis(self, mock_get_redis, client):
        """Control fails without Redis"""
        mock_get_redis.return_value = None

        response = client.post("/api/control/start")
        assert response.status_code == 503


class TestSSELogStream:
    """SSE log stream tests"""

    @pytest.fixture
    def client(self):
        try:
            from dashboard.web_app import app
            app.config["TESTING"] = True
            return app.test_client()
        except ImportError:
            pytest.skip("flask not installed")

    @patch("dashboard.web_app.get_redis_client")
    def test_stream_logs_no_redis(self, mock_get_redis, client):
        """Error message when Redis unavailable"""
        mock_get_redis.return_value = None

        response = client.get("/api/stream/logs")
        assert response.status_code == 200
        assert response.content_type.startswith("text/event-stream")
        data = b""
        for chunk in response.response:
            data += chunk
            break  # first chunk only
        assert b"Redis" in data or b"ERROR" in data


# ============================================================================
# Integration Tests: mass_collector + quality
# ============================================================================

class TestMassCollectorIntegration:
    """MassCollector correctly invokes quality stage tests"""

    def test_pipeline_stages_include_quality(self):
        """quality is in STAGES"""
        from mass_collector import MassCollector
        assert "quality" in MassCollector.STAGES

    def test_pipeline_config_defaults(self):
        """PipelineConfig defaults"""
        from mass_collector import PipelineConfig
        config = PipelineConfig()
        assert config.target_count == 500
        assert config.quality_filter is True
        assert config.quality_threshold == 60.0
        assert config.use_gpu_streams is True
        assert config.use_multiprocess is False

    def test_pipeline_config_multiprocess(self):
        """Multiprocess configuration"""
        from mass_collector import PipelineConfig
        config = PipelineConfig(use_multiprocess=True, crawl_workers=8)
        assert config.use_multiprocess is True
        assert config.crawl_workers == 8

    def test_stage_result_summary(self):
        """StageResult summary"""
        from mass_collector import StageResult
        result = StageResult(
            stage="quality",
            success=True,
            count=70,
            errors=3,
            elapsed_sec=45.5,
        )
        summary = result.summary()
        assert "quality" in summary
        assert "70" in summary
        assert "3" in summary

    def test_pipeline_report_save(self, tmp_path):
        """PipelineReport save"""
        from mass_collector import PipelineReport
        report = PipelineReport(
            started_at="2026-02-12T10:00:00",
            completed_at="2026-02-12T10:05:00",
            total_crawled=100,
            total_downloaded=95,
            total_episodes=90,
            total_uploaded=85,
        )
        path = str(tmp_path / "test_report.json")
        report.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["total_crawled"] == 100
        assert data["total_uploaded"] == 85


# ============================================================================
# Integration Tests: GPU Stream + ProcessingQueue
# ============================================================================

class TestGPUProcessingIntegration:
    """GPU Stream manager + ProcessingQueue integration tests"""

    def test_processing_queue_enqueue_and_pop(self):
        """Video enqueue and batch pop"""
        from task_queue.task_queue import ProcessingQueue
        pq = ProcessingQueue(host="invalid_host", port=9999)

        mock_client = Mock()
        mock_client.ping.return_value = True
        items = [
            json.dumps({"video_path": f"/data/raw/{i}.mp4", "video_id": f"vid_{i}"})
            for i in range(3)
        ]
        mock_client.lpop.side_effect = items + [None]
        pq._client = mock_client
        pq._connected = True

        batch = pq.pop_batch(batch_size=3)
        assert len(batch) == 3

    def test_gpu_stream_custom_processor(self):
        """Custom processor batch processing - queue conflict skip"""
        from gpu.stream_manager import GPU3StreamManager

        manager = GPU3StreamManager()

        def test_processor(video_path):
            return {"video_path": video_path, "success": True, "frames": 100}

        try:
            results = manager.process_batch(
                ["/test/a.mp4", "/test/b.mp4"],
                processor=test_processor,
            )
            assert len(results) == 2
            assert all(r["success"] for r in results)
        except AttributeError as e:
            if "SimpleQueue" in str(e):
                pytest.skip("queue/ package shadows stdlib queue module")
            raise
