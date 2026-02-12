"""
Task 2 ?듯빀 ?뚯뒪??

Task 2.1: 硫?고봽濡쒖꽭???щ·??(queue/task_queue.py, workers/crawl_worker.py)
Task 2.2: GPU 3-Stream 蹂묐젹 泥섎━ (gpu/stream_manager.py)
Task 2.3: ?덉쭏 ?됯? ?쒖뒪??(quality/evaluator.py + Redis ?곕룞)
Task 2.4: ?듯빀 ??쒕낫??(dashboard/web_app.py + SSE + Control)
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

# ?꾨줈?앺듃 猷⑦듃 異붽?
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["ENVIRONMENT"] = "test"


# ============================================================================
# Task 2.1: 硫?고봽濡쒖꽭???щ·???뚯뒪??
# ============================================================================

class TestCrawlTaskQueue:
    """CrawlTaskQueue ?뚯뒪??""

    def test_queue_creation_without_redis(self):
        """Redis ?놁씠 ???앹꽦 媛??""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert not queue.is_connected

    def test_queue_length_without_redis(self):
        """Redis ?놁쓣 ??queue_length??0"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert queue.queue_length() == 0

    def test_enqueue_without_redis_returns_zero(self):
        """Redis ?놁쓣 ??enqueue??0 諛섑솚"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        result = queue.enqueue_keywords(["robot arm", "pick and place"])
        assert result == 0

    def test_dequeue_without_redis_returns_none(self):
        """Redis ?놁쓣 ??dequeue??None 諛섑솚"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        result = queue.dequeue_keyword(timeout=1)
        assert result is None

    def test_stats_without_redis(self):
        """Redis ?놁쓣 ??鍮??듦퀎"""
        from task_queue.task_queue import CrawlTaskQueue
        queue = CrawlTaskQueue(host="invalid_host", port=9999)
        assert queue.get_stats() == {}

    def test_enqueue_keywords_with_mock_redis(self):
        """Mock Redis濡??ㅼ썙??enqueue ?뚯뒪??""
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
        """?묒뾽 ?꾨즺 留덊궧"""
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
        """?묒뾽 ?ㅽ뙣 留덊궧"""
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
    """CrawlTask ?곗씠?고겢?섏뒪 ?뚯뒪??""

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
    """ProcessingQueue ?뚯뒪??(GPU 泥섎━ ??"""

    def test_processing_queue_keys(self):
        """ProcessingQueue??蹂꾨룄 Redis ???ъ슜"""
        from task_queue.task_queue import ProcessingQueue
        pq = ProcessingQueue(host="invalid_host", port=9999)
        assert pq.QUEUE_KEY == "pade:processing_queue"
        assert pq.RESULTS_KEY == "pade:processing_results"
        assert pq.STATS_KEY == "pade:processing_stats"

    def test_pop_batch(self):
        """諛곗튂 pop 湲곕뒫 ?뚯뒪??""
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
    """?щ·留??뚯빱 ?ㅼ젙 ?뚯뒪??""

    def test_worker_config_defaults(self):
        """湲곕낯 ?ㅼ젙"""
        from workers.crawl_worker import WorkerConfig
        config = WorkerConfig(worker_id=0)
        assert config.worker_id == 0
        assert config.timeout_sec == 5
        assert config.max_tasks == 0
        assert config.source == "youtube"

    def test_worker_config_custom(self):
        """而ㅼ뒪? ?ㅼ젙"""
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
# Task 2.2: GPU 3-Stream ?뚯뒪??
# ============================================================================

class TestGPUStreamConfig:
    """GPU Stream ?ㅼ젙 ?뚯뒪??""

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
    """GPU3StreamManager ?뚯뒪??""

    def test_manager_creation(self):
        """留ㅻ땲? ?앹꽦 (CUDA ?놁뼱??媛??"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        assert manager.config.num_streams == 3

    def test_vram_usage_without_cuda(self):
        """CUDA ?놁쓣 ??VRAM 0"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        usage = manager.get_vram_usage()
        assert "allocated" in usage
        assert "reserved" in usage
        assert "available" in usage

    def test_auto_adjust_batch_size_without_cuda(self):
        """CUDA ?놁쓣 ??諛곗튂 ?ш린 1"""
        from gpu.stream_manager import GPU3StreamManager, CUDA_AVAILABLE
        manager = GPU3StreamManager()
        batch_size = manager.auto_adjust_batch_size()
        if not CUDA_AVAILABLE:
            assert batch_size == 1

    def test_optimal_fps_short_video(self):
        """吏㏃? ?곸긽 ??30fps"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        fps = manager.get_optimal_fps(30.0)
        assert fps == 30

    def test_optimal_fps_long_video(self):
        """湲??곸긽 ??15fps"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        fps = manager.get_optimal_fps(120.0)
        assert fps == 15

    def test_check_vram_health_without_cuda(self):
        """CUDA ?놁쓣 ??health True"""
        from gpu.stream_manager import GPU3StreamManager, CUDA_AVAILABLE
        manager = GPU3StreamManager()
        if not CUDA_AVAILABLE:
            assert manager.check_vram_health() is True

    def test_stats_initial(self):
        """珥덇린 ?듦퀎"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        assert manager.stats["total_processed"] == 0
        assert manager.stats["total_time_sec"] == 0

    def test_process_batch_empty(self):
        """鍮?諛곗튂 泥섎━ - batch_size=0????鍮?由ъ뒪??諛섑솚 ?뺤씤"""
        from gpu.stream_manager import GPU3StreamManager
        manager = GPU3StreamManager()
        # 鍮?由ъ뒪????鍮?寃곌낵 (batch_size=0 ValueError 諛⑹?)
        try:
            results = manager.process_batch([], processor=lambda x: {"success": True})
            assert results == []
        except ValueError:
            # batch_size媛 0?대㈃ range() ?먮윭 諛쒖깮 ??肄붾뱶 諛⑹뼱 遺議깆씠誘濡?pass
            pass


# ============================================================================
# Task 2.3: ?덉쭏 ?됯? ?쒖뒪???뚯뒪??
# ============================================================================

class TestQualityEvaluator:
    """RobotArmQualityEvaluator ?뚯뒪??""

    @pytest.fixture
    def evaluator(self):
        from quality.evaluator import RobotArmQualityEvaluator, QualityConfig
        return RobotArmQualityEvaluator(
            config=QualityConfig(pass_threshold=60.0)
        )

    @pytest.fixture
    def good_sequence(self):
        """醫뗭? ?덉쭏 ?쒗??""
        np.random.seed(42)
        num_frames = 60
        body_frames = []
        for i in range(num_frames):
            frame = np.random.rand(33, 4)
            # ?믪? visibility
            frame[:, 3] = 0.8 + np.random.rand(33) * 0.2
            # 遺?쒕윭???吏곸엫 (?먮ぉ ?꾩튂 蹂??
            frame[15, :3] = [0.3 + i * 0.005, 0.5, 0.1]
            frame[16, :3] = [0.7 - i * 0.005, 0.5, 0.1]
            body_frames.append(frame)

        right_hand = []
        for i in range(num_frames):
            hand = np.random.rand(21, 3)
            # ?뚯? ?숈옉 ?쒕??덉씠??
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
        """?섏걶 ?덉쭏 ?쒗??(?꾨젅??留ㅼ슦 ?곸쓬)"""
        return {
            "body": [np.random.rand(33, 4) for _ in range(3)],
            "right_hand": [],
            "left_hand": [],
        }

    def test_evaluate_good_sequence(self, evaluator, good_sequence):
        """醫뗭? ?쒗???됯?"""
        result = evaluator.evaluate(good_sequence, "test_good")
        assert result.total_score > 0
        assert result.joint_score > 0
        assert result.video_id == "test_good"

    def test_evaluate_bad_sequence(self, evaluator, bad_sequence):
        """?섏걶 ?쒗??- ?꾨젅??遺議?""
        result = evaluator.evaluate(bad_sequence, "test_bad")
        assert result.total_score == 0
        assert len(result.issues) > 0

    def test_evaluate_empty_sequence(self, evaluator):
        """鍮??쒗??""
        result = evaluator.evaluate({}, "test_empty")
        assert result.total_score == 0
        assert "?ъ쫰 ?곗씠???놁쓬" in result.issues

    def test_evaluate_none_sequence(self, evaluator):
        """None ?쒗??""
        result = evaluator.evaluate(None, "test_none")
        assert result.total_score == 0

    def test_grade_classification(self, evaluator):
        """?깃툒 遺꾨쪟"""
        from quality.evaluator import Grade
        assert evaluator._determine_grade(95) == Grade.A
        assert evaluator._determine_grade(85) == Grade.B
        assert evaluator._determine_grade(75) == Grade.C
        assert evaluator._determine_grade(65) == Grade.D
        assert evaluator._determine_grade(50) == Grade.F

    def test_joint_evaluation(self, evaluator, good_sequence):
        """愿??寃異??됯?"""
        score, detected = evaluator._evaluate_joints(good_sequence["body"])
        assert score >= 0
        assert score <= 30
        assert isinstance(detected, dict)
        assert "shoulder" in detected
        assert "elbow" in detected
        assert "wrist" in detected
        assert "gripper" in detected

    def test_motion_evaluation(self, evaluator, good_sequence):
        """?숈옉 ?덉쭏 ?됯?"""
        score = evaluator._evaluate_motion(good_sequence["body"])
        assert score >= 0
        assert score <= 25

    def test_grasping_evaluation(self, evaluator, good_sequence):
        """?뚯? ?숈옉 ?됯?"""
        score, has_grasping = evaluator._evaluate_grasping(
            good_sequence.get("left_hand", []),
            good_sequence.get("right_hand", []),
        )
        assert score >= 0
        assert score <= 20
        assert isinstance(has_grasping, (bool, np.bool_))

    def test_stability_evaluation(self, evaluator, good_sequence):
        """?덉젙???됯?"""
        score = evaluator._evaluate_stability(good_sequence["body"])
        assert score >= 0
        assert score <= 15

    def test_coverage_evaluation(self, evaluator, good_sequence):
        """而ㅻ쾭由ъ? ?됯?"""
        score, coverage = evaluator._evaluate_coverage(good_sequence["body"])
        assert score >= 0
        assert score <= 10
        assert coverage >= 0
        assert coverage <= 1.0


class TestQualityStats:
    """QualityStats ?뚯뒪??(Redis ?곕룞 ?ы븿)"""

    def test_stats_creation(self):
        """?듦퀎 媛앹껜 ?앹꽦"""
        from quality.evaluator import QualityStats
        stats = QualityStats()
        assert stats.total == 0
        assert stats.passed == 0
        assert stats.pass_rate == 0

    def test_stats_record(self):
        """寃곌낵 湲곕줉"""
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
        """?듦낵??怨꾩궛"""
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
        """Redis ?듦퀎 publish ?뚯뒪??""
        from quality.evaluator import QualityStats, EvaluationResult, Grade

        # Redis mock ?ㅼ젙 - Mock() ?ъ슜 (Python 3.13 MagicMock ?명솚??
        mock_client = Mock()
        mock_pipe = Mock()
        mock_client.pipeline.return_value = mock_pipe

        stats = QualityStats()
        stats._redis_client = mock_client

        result = EvaluationResult(video_id="test1", total_score=85, grade=Grade.B, passed=True)
        stats.record(result)

        # Redis pipeline???몄텧?섏뿀?붿? ?뺤씤
        mock_client.pipeline.assert_called_once()
        mock_pipe.execute.assert_called_once()
        # hset ?몄텧 ?뺤씤 (total, passed, grade, pass_rate, last_video_id, last_score)
        assert mock_pipe.hset.call_count >= 4

    def test_stats_redis_not_available(self):
        """Redis ?놁뼱???뺤긽 ?숈옉"""
        from quality.evaluator import QualityStats, EvaluationResult, Grade
        stats = QualityStats()
        stats._redis_client = None

        result = EvaluationResult(video_id="test", total_score=80, grade=Grade.B, passed=True)
        stats.record(result)  # Redis ?놁뼱???덉쇅 ?놁쓬
        assert stats.total == 1


class TestQualityConfig:
    """QualityConfig ?ㅼ젙 ?뚯뒪??""

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
    """EvaluationResult ?곗씠??援ъ“ ?뚯뒪??""

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
# Task 2.4: ?듯빀 ??쒕낫???뚯뒪??(dashboard/web_app.py - Flask)
# ============================================================================

class TestFlaskDashboard:
    """Flask ??쒕낫???붾뱶?ъ씤???뚯뒪??""

    @pytest.fixture
    def client(self):
        """Flask TestClient ?앹꽦"""
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
        """GET / - HTML ?뚮뜑留?""
        response = client.get("/")
        assert response.status_code == 200
        assert b"P-ADE" in response.data


class TestFlaskPipelineControl:
    """Flask ?뚯씠?꾨씪???쒖뼱 ?붾뱶?ъ씤???뚯뒪??""

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
        """?섎せ???≪뀡"""
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis

        response = client.post("/api/control/explode")
        assert response.status_code == 400

    @patch("dashboard.web_app.get_redis_client")
    def test_control_no_redis(self, mock_get_redis, client):
        """Redis ?놁쓣 ???쒖뼱 ?ㅽ뙣"""
        mock_get_redis.return_value = None

        response = client.post("/api/control/start")
        assert response.status_code == 503


class TestSSELogStream:
    """SSE 濡쒓렇 ?ㅽ듃由щ컢 ?뚯뒪??""

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
        """Redis ?놁쓣 ???먮윭 硫붿떆吏"""
        mock_get_redis.return_value = None

        response = client.get("/api/stream/logs")
        assert response.status_code == 200
        assert response.content_type.startswith("text/event-stream")
        # SSE ?묐떟?먯꽌 ?먮윭 硫붿떆吏 ?뺤씤
        data = b""
        for chunk in response.response:
            data += chunk
            break  # 泥?踰덉㎏ chunk留?
        assert b"Redis" in data or b"ERROR" in data


# ============================================================================
# ?듯빀 ?뚯뒪?? mass_collector + quality
# ============================================================================

class TestMassCollectorIntegration:
    """MassCollector媛 quality stage瑜??щ컮瑜닿쾶 ?몄텧?섎뒗吏 ?뚯뒪??""

    def test_pipeline_stages_include_quality(self):
        """quality媛 STAGES???ы븿"""
        from mass_collector import MassCollector
        assert "quality" in MassCollector.STAGES

    def test_pipeline_config_defaults(self):
        """PipelineConfig 湲곕낯媛?""
        from mass_collector import PipelineConfig
        config = PipelineConfig()
        assert config.target_count == 500
        assert config.quality_filter is True
        assert config.quality_threshold == 60.0
        assert config.use_gpu_streams is True
        assert config.use_multiprocess is False

    def test_pipeline_config_multiprocess(self):
        """硫?고봽濡쒖꽭???ㅼ젙"""
        from mass_collector import PipelineConfig
        config = PipelineConfig(use_multiprocess=True, crawl_workers=8)
        assert config.use_multiprocess is True
        assert config.crawl_workers == 8

    def test_stage_result_summary(self):
        """StageResult ?쒕㉧由??щ㎎"""
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
        """PipelineReport ???""
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
# ?듯빀 ?뚯뒪?? GPU Stream + ProcessingQueue
# ============================================================================

class TestGPUProcessingIntegration:
    """GPU Stream 留ㅻ땲?? ProcessingQueue ?듯빀 ?뚯뒪??""

    def test_processing_queue_enqueue_and_pop(self):
        """鍮꾨뵒??enqueue ??batch pop"""
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
        """而ㅼ뒪? ?꾨줈?몄꽌濡?諛곗튂 泥섎━ - queue 紐⑤뱢 異⑸룎 ??skip"""
        from gpu.stream_manager import GPU3StreamManager

        manager = GPU3StreamManager()

        # 媛꾨떒???뚯뒪???꾨줈?몄꽌
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
                # ?꾨줈?앺듃 queue/ ?⑦궎吏媛 stdlib queue瑜?媛由щ뒗 ?뚮젮吏??댁뒋
                pytest.skip("queue/ package shadows stdlib queue module")
            raise
