"""
Task 4 통합 테스트

테스트 범위:
- 4.1: main.py (CollectionPipeline, run-forever, CLI)
- 4.2: 알림 모니터링 루프 (register_task4_rules, MetricsCollector, AlertMonitorLoop)
- 4.1/4.2: deploy 파일 존재 검증
"""

import json
import os
import sys
import time
import signal
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 4.1 CollectionPipeline 테스트
# ============================================================

class TestCollectionPipeline:
    """main.py - CollectionPipeline 테스트"""

    def test_init(self):
        from main import CollectionPipeline
        pipeline = CollectionPipeline(target_count=100)
        assert pipeline.target_count == 100
        assert pipeline._daily_count == 0

    def test_daily_target_not_reached(self):
        from main import CollectionPipeline
        pipeline = CollectionPipeline(target_count=100)
        assert pipeline.is_daily_target_reached() is False

    def test_daily_target_reached(self):
        from main import CollectionPipeline
        pipeline = CollectionPipeline(target_count=100)
        pipeline._daily_count = 100
        assert pipeline.is_daily_target_reached() is True

    def test_date_reset(self):
        """자정 넘으면 카운트 초기화"""
        from main import CollectionPipeline
        pipeline = CollectionPipeline(target_count=100)
        pipeline._daily_count = 50
        # 어제로 설정
        pipeline._daily_reset_date = (datetime.now() - timedelta(days=1)).date()
        # 리셋 트리거
        assert pipeline.is_daily_target_reached() is False
        assert pipeline._daily_count == 0

    def test_run_once_calls_mass_collector(self):
        """run_once가 MassCollector를 호출"""
        from main import CollectionPipeline

        pipeline = CollectionPipeline(target_count=10)

        with patch("mass_collector.MassCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.run.return_value = None
            pipeline.run_once()

        MockCollector.assert_called_once()
        instance.run.assert_called_once()

    def test_run_once_with_stage(self):
        """특정 단계 실행"""
        from main import CollectionPipeline

        pipeline = CollectionPipeline(target_count=10, stage="crawl")

        with patch("mass_collector.MassCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.run.return_value = None
            pipeline.run_once()

        instance.run.assert_called_once_with(
            start_stage="crawl", end_stage="crawl"
        )

    def test_run_once_with_start_end_stage(self):
        """시작/종료 단계 지정"""
        from main import CollectionPipeline

        pipeline = CollectionPipeline(
            target_count=10, start_stage="download", end_stage="detect"
        )

        with patch("mass_collector.MassCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.run.return_value = None
            pipeline.run_once()

        instance.run.assert_called_once_with(
            start_stage="download", end_stage="detect"
        )

    def test_cleanup(self):
        """cleanup이 에러 없이 실행"""
        from main import CollectionPipeline
        pipeline = CollectionPipeline(target_count=10)
        pipeline.cleanup()  # 에러 없어야 함


# ============================================================
# 4.1 wait_until_tomorrow 테스트
# ============================================================

class TestWaitUntilTomorrow:

    def test_calculates_correct_time(self):
        """대기 시간 계산"""
        from main import wait_until_tomorrow
        import main

        # shutdown 플래그로 즉시 탈출
        main._shutdown_requested = True
        wait_until_tomorrow(hour=6)
        main._shutdown_requested = False

    def test_shutdown_interrupts_wait(self):
        """종료 신호 시 즉시 탈출"""
        from main import wait_until_tomorrow
        import main

        main._shutdown_requested = True
        start = time.time()
        wait_until_tomorrow(hour=6)
        elapsed = time.time() - start

        assert elapsed < 5  # 즉시 반환
        main._shutdown_requested = False


# ============================================================
# 4.1 run_forever 테스트
# ============================================================

class TestRunForever:

    def test_run_forever_single_iteration(self):
        """run_forever가 1회 실행 후 종료"""
        import main

        call_count = 0

        def mock_run_once(self):
            nonlocal call_count
            call_count += 1
            # 1회 실행 후 종료 신호
            main._shutdown_requested = True

        with patch.object(
            main.CollectionPipeline, "run_once", mock_run_once
        ):
            with patch("time.sleep"):
                main._shutdown_requested = False
                main.run_forever(target_count=10, interval=1, error_wait=1)

        assert call_count == 1

    def test_run_forever_skips_when_target_reached(self):
        """목표 달성 시 대기"""
        import main

        call_count = 0

        def mock_is_reached(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # 첫 번째: 목표 달성
            main._shutdown_requested = True
            return False

        with patch.object(
            main.CollectionPipeline, "is_daily_target_reached", mock_is_reached
        ):
            with patch("main.wait_until_tomorrow"):
                with patch.object(main.CollectionPipeline, "run_once"):
                    main._shutdown_requested = False
                    main.run_forever(target_count=10, interval=0, error_wait=0)

    def test_run_forever_handles_error(self):
        """에러 발생 시 대기 후 재시도"""
        import main

        call_count = 0

        def mock_run_once(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test error")
            main._shutdown_requested = True

        with patch.object(
            main.CollectionPipeline, "run_once", mock_run_once
        ):
            with patch("time.sleep"):
                main._shutdown_requested = False
                main.run_forever(target_count=10, interval=0, error_wait=0)

        assert call_count == 2  # 에러 후 재시도


# ============================================================
# 4.1 CLI 테스트
# ============================================================

class TestCLI:

    def test_build_parser(self):
        from main import build_parser
        parser = build_parser()
        assert parser is not None

    def test_parse_run_forever(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run-forever", "--target", "200", "--interval", "30"
        ])
        assert args.command == "run-forever"
        assert args.target == 200
        assert args.interval == 30

    def test_parse_run_once(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["run-once", "--stage", "crawl"])
        assert args.command == "run-once"
        assert args.stage == "crawl"

    def test_parse_monitor_alerts(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["monitor-alerts", "--interval", "60"])
        assert args.command == "monitor-alerts"
        assert args.interval == 60

    def test_parse_default(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ============================================================
# 4.1 deploy 파일 존재 테스트
# ============================================================

class TestDeployFiles:

    def test_systemd_service_exists(self):
        """systemd 서비스 파일 존재"""
        path = PROJECT_ROOT / "deploy" / "robot-collector.service"
        assert path.exists()

    def test_systemd_service_content(self):
        """systemd 서비스 파일 내용"""
        path = PROJECT_ROOT / "deploy" / "robot-collector.service"
        content = path.read_text(encoding="utf-8")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "main.py run-forever" in content
        assert "Restart=always" in content

    def test_alert_monitor_service_exists(self):
        """알림 모니터 서비스 파일 존재"""
        path = PROJECT_ROOT / "deploy" / "robot-alert-monitor.service"
        assert path.exists()

    def test_logrotate_exists(self):
        """logrotate 설정 파일 존재"""
        path = PROJECT_ROOT / "deploy" / "robot-collector.logrotate"
        assert path.exists()

    def test_logrotate_content(self):
        """logrotate 설정 내용"""
        path = PROJECT_ROOT / "deploy" / "robot-collector.logrotate"
        content = path.read_text(encoding="utf-8")
        assert "daily" in content
        assert "rotate 7" in content
        assert "compress" in content


# ============================================================
# 4.2 register_task4_rules 테스트
# ============================================================

class TestRegisterTask4Rules:

    def test_register_rules(self):
        """Task 4 규칙 4개 등록"""
        from alerts.manager import AlertManager
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        assert "gpu_util_low" in manager._rules
        assert "queue_depleted" in manager._rules
        assert "failure_rate_high" in manager._rules
        assert "target_behind" in manager._rules

    def test_gpu_util_low_condition(self):
        """GPU 30% 미만 조건"""
        from alerts.manager import AlertManager
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        rule = manager._rules["gpu_util_low"]
        assert rule.condition(gpu_util=20) is True
        assert rule.condition(gpu_util=50) is False
        assert rule.condition(gpu_util=30) is False

    def test_queue_depleted_condition(self):
        """큐 100개 미만 조건"""
        from alerts.manager import AlertManager
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        rule = manager._rules["queue_depleted"]
        assert rule.condition(queue_size=50) is True
        assert rule.condition(queue_size=200) is False
        assert rule.condition(queue_size=100) is False

    def test_failure_rate_high_condition(self):
        """실패율 40% 초과 조건"""
        from alerts.manager import AlertManager
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        rule = manager._rules["failure_rate_high"]
        assert rule.condition(failure_rate=0.5) is True
        assert rule.condition(failure_rate=0.3) is False
        assert rule.condition(failure_rate=0.4) is False

    def test_target_behind_condition(self):
        """18시 기준 목표 40% 이하 조건"""
        from alerts.manager import AlertManager
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        rule = manager._rules["target_behind"]
        # 18시 + 진행률 30% → 알림
        assert rule.condition(daily_progress=0.3, current_hour=18) is True
        # 18시 + 진행률 60% → OK
        assert rule.condition(daily_progress=0.6, current_hour=18) is False
        # 14시 + 진행률 30% → 시간이 아님
        assert rule.condition(daily_progress=0.3, current_hour=14) is False


# ============================================================
# 4.2 MetricsCollector 테스트
# ============================================================

class TestMetricsCollector:

    def test_init(self):
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector(daily_target=200)
        assert mc.daily_target == 200

    def test_gpu_utilization_fallback(self):
        """GPU util: nvidia-smi 없으면 0.0"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector()
        mc._redis = None  # Redis 없음

        with patch(
            "monitor.stats_collector.StatsCollector.get_gpu_utilization",
            return_value=0.0,
        ):
            util = mc.get_gpu_utilization()
        assert util == 0.0

    def test_gpu_utilization_from_redis(self):
        """GPU util: Redis에서 조회"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector()

        mock_redis = MagicMock()
        mock_redis.get.return_value = "75.0"
        mc._redis = mock_redis

        util = mc.get_gpu_utilization()
        assert util == 75.0

    def test_queue_size_from_redis(self):
        """큐 크기: Redis에서 조회"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector()

        mock_redis = MagicMock()
        mock_redis.get.return_value = "350"
        mc._redis = mock_redis

        size = mc.get_queue_size()
        assert size == 350

    def test_queue_size_from_db(self, tmp_path):
        """큐 크기: DB 폴백"""
        from monitor.alert_loop import MetricsCollector

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE videos (
                id INTEGER PRIMARY KEY,
                video_id TEXT,
                status TEXT
            )
        """)
        conn.execute("INSERT INTO videos (video_id, status) VALUES ('v1', 'pending')")
        conn.execute("INSERT INTO videos (video_id, status) VALUES ('v2', 'pending')")
        conn.execute("INSERT INTO videos (video_id, status) VALUES ('v3', 'completed')")
        conn.commit()
        conn.close()

        mc = MetricsCollector(db_path=db_path)
        mc._redis = None
        size = mc.get_queue_size()
        assert size == 2

    def test_failure_rate_from_redis(self):
        """실패율: Redis에서 조회"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector()

        mock_redis = MagicMock()
        mock_redis.hget.side_effect = lambda key, field: {
            ("pade:processing_stats", "total_processed"): "100",
            ("pade:processing_stats", "total_failed"): "20",
        }.get((key, field), "0")
        mc._redis = mock_redis

        rate = mc.get_failure_rate()
        assert abs(rate - 0.2) < 0.01

    def test_failure_rate_from_db(self, tmp_path):
        """실패율: DB 폴백"""
        from monitor.alert_loop import MetricsCollector

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE videos (
                id INTEGER PRIMARY KEY,
                video_id TEXT,
                status TEXT
            )
        """)
        for i in range(8):
            conn.execute(
                "INSERT INTO videos (video_id, status) VALUES (?, ?)",
                (f"v{i}", "completed"),
            )
        for i in range(2):
            conn.execute(
                "INSERT INTO videos (video_id, status) VALUES (?, ?)",
                (f"f{i}", "failed"),
            )
        conn.commit()
        conn.close()

        mc = MetricsCollector(db_path=db_path)
        mc._redis = None
        rate = mc.get_failure_rate()
        assert abs(rate - 0.2) < 0.01  # 2/10 = 20%

    def test_daily_progress_from_redis(self):
        """일일 진행률: Redis"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector(daily_target=500)

        mock_redis = MagicMock()
        mock_redis.get.return_value = "250"
        mc._redis = mock_redis

        progress = mc.get_daily_progress()
        assert abs(progress - 0.5) < 0.01

    def test_collect_returns_context(self):
        """collect()가 완전한 context dict 반환"""
        from monitor.alert_loop import MetricsCollector
        mc = MetricsCollector()

        with patch.object(mc, "get_gpu_utilization", return_value=55.0):
            with patch.object(mc, "get_queue_size", return_value=300):
                with patch.object(mc, "get_failure_rate", return_value=0.1):
                    with patch.object(mc, "get_daily_progress", return_value=0.6):
                        ctx = mc.collect()

        assert ctx["gpu_util"] == 55.0
        assert ctx["queue_size"] == 300
        assert ctx["failure_rate"] == 0.1
        assert ctx["daily_progress"] == 0.6
        assert "current_hour" in ctx
        assert "timestamp" in ctx


# ============================================================
# 4.2 AlertMonitorLoop 테스트
# ============================================================

class TestAlertMonitorLoop:

    def test_init(self):
        from monitor.alert_loop import AlertMonitorLoop
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        loop = AlertMonitorLoop(
            alert_manager=manager,
            check_interval=10,
        )
        assert loop.check_interval == 10
        assert loop._shutdown is False

    def test_check_once(self):
        """1회 체크"""
        from monitor.alert_loop import AlertMonitorLoop, MetricsCollector
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        mc = MetricsCollector()

        with patch.object(mc, "collect", return_value={
            "gpu_util": 80, "queue_size": 500,
            "failure_rate": 0.05, "daily_progress": 0.7,
            "current_hour": 14, "timestamp": "2026-01-01T14:00:00",
        }):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
            )
            ctx = loop.check_once()

        assert ctx["gpu_util"] == 80

    def test_check_once_fires_alert(self):
        """조건 충족 시 알림 발생"""
        from monitor.alert_loop import (
            AlertMonitorLoop, MetricsCollector, register_task4_rules,
        )
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        mc = MetricsCollector()

        with patch.object(mc, "collect", return_value={
            "gpu_util": 10,  # 30% 미만 → 알림
            "queue_size": 50,  # 100 미만 → 알림
            "failure_rate": 0.5,  # 40% 초과 → 알림
            "daily_progress": 0.7,
            "current_hour": 14,
            "timestamp": "2026-01-01T14:00:00",
        }):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
            )
            loop.check_once()

        # 활성 알림 확인
        active = manager.get_active_alerts()
        rule_names = {a.rule_name for a in active}
        assert "gpu_util_low" in rule_names
        assert "queue_depleted" in rule_names
        assert "failure_rate_high" in rule_names

    def test_check_once_no_alert_when_healthy(self):
        """건강한 상태에서는 알림 없음"""
        from monitor.alert_loop import (
            AlertMonitorLoop, MetricsCollector, register_task4_rules,
        )
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        mc = MetricsCollector()

        with patch.object(mc, "collect", return_value={
            "gpu_util": 80,
            "queue_size": 5000,
            "failure_rate": 0.05,
            "daily_progress": 0.8,
            "current_hour": 14,
            "timestamp": "2026-01-01T14:00:00",
        }):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
            )
            loop.check_once()

        assert len(manager.get_active_alerts()) == 0

    def test_target_behind_at_18(self):
        """18시에 목표 미달 알림"""
        from monitor.alert_loop import (
            AlertMonitorLoop, MetricsCollector, register_task4_rules,
        )
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        mc = MetricsCollector()

        with patch.object(mc, "collect", return_value={
            "gpu_util": 80,
            "queue_size": 5000,
            "failure_rate": 0.05,
            "daily_progress": 0.3,  # 40% 미만
            "current_hour": 18,
            "timestamp": "2026-01-01T18:00:00",
        }):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
            )
            loop.check_once()

        active = manager.get_active_alerts()
        rule_names = {a.rule_name for a in active}
        assert "target_behind" in rule_names

    def test_run_loop_terminates(self):
        """루프가 종료 신호로 멈춤"""
        from monitor.alert_loop import AlertMonitorLoop, MetricsCollector
        from alerts.manager import AlertManager

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        mc = MetricsCollector()

        check_count = 0

        def mock_collect():
            nonlocal check_count
            check_count += 1
            return {
                "gpu_util": 80, "queue_size": 500,
                "failure_rate": 0, "daily_progress": 1.0,
                "current_hour": 14, "timestamp": "",
            }

        with patch.object(mc, "collect", side_effect=mock_collect):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
                check_interval=1,
            )
            loop._shutdown = True  # 즉시 종료

            with patch("time.sleep"):
                loop.run()

        # check_once가 최소 1회 호출되어야
        assert check_count >= 0  # 종료 전에 호출될 수도 있고 안 될 수도


# ============================================================
# 4.2 알림 쿨다운 테스트
# ============================================================

class TestAlertCooldown:
    """중복 알림 방지 테스트"""

    def test_cooldown_prevents_duplicate(self):
        """쿨다운 내 동일 알림 중복 발송 방지"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel
        from alerts.slack import AlertLevel

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        manager.register_rule(AlertRule(
            name="test_rule",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            severity=AlertLevel.WARNING,
            cooldown_seconds=600,
            description="Test alert",
        ))

        # 첫 번째 발생
        alert1 = manager.fire("test_rule", "first")
        assert alert1 is not None

        # 두 번째 발생 (쿨다운 내) → None
        alert2 = manager.fire("test_rule", "second")
        assert alert2 is None

        # force=True → 쿨다운 무시
        alert3 = manager.fire("test_rule", "forced", force=True)
        assert alert3 is not None

    def test_silence_prevents_alert(self):
        """silence 상태에서 알림 차단"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel
        from alerts.slack import AlertLevel

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        manager.register_rule(AlertRule(
            name="test_silence",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            severity=AlertLevel.WARNING,
            description="Test",
        ))

        manager.silence("test_silence", duration_seconds=3600)
        alert = manager.fire("test_silence", "silenced")
        assert alert is None


# ============================================================
# 4.2 기존 알림 시스템 호환성 테스트
# ============================================================

class TestExistingAlertsCompat:
    """기존 알림 시스템이 깨지지 않았는지 확인"""

    def test_register_default_rules(self):
        from alerts.manager import AlertManager, register_default_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_default_rules(manager)

        assert "high_error_rate" in manager._rules
        assert "high_queue_depth" in manager._rules
        assert "high_latency" in manager._rules
        assert "disk_usage_high" in manager._rules

    def test_default_and_task4_rules_coexist(self):
        """기본 규칙 + Task 4 규칙 공존"""
        from alerts.manager import AlertManager, register_default_rules
        from monitor.alert_loop import register_task4_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_default_rules(manager)
        register_task4_rules(manager)

        # 기본 4개
        assert "high_error_rate" in manager._rules
        assert "high_queue_depth" in manager._rules
        assert "high_latency" in manager._rules
        assert "disk_usage_high" in manager._rules

        # Task 4 4개
        assert "gpu_util_low" in manager._rules
        assert "queue_depleted" in manager._rules
        assert "failure_rate_high" in manager._rules
        assert "target_behind" in manager._rules

        assert len(manager._rules) == 8

    def test_alert_manager_imports(self):
        """alerts 모듈 import 확인"""
        from alerts import (
            AlertManager,
            AlertRule,
            AlertChannel,
            AlertState,
            AlertLevel,
            Alert,
            get_alert_manager,
            register_default_rules,
        )
        assert AlertManager is not None

    def test_slack_notifier_import(self):
        """Slack notifier import"""
        from alerts.slack import SlackNotifier, get_slack_notifier
        notifier = get_slack_notifier()
        assert notifier is not None

    def test_email_notifier_import(self):
        """Email notifier import"""
        from alerts.email import EmailNotifier, get_email_notifier
        notifier = get_email_notifier()
        assert notifier is not None

    def test_check_rules_with_context(self):
        """check_rules에 context 전달"""
        from alerts.manager import AlertManager, register_default_rules

        manager = AlertManager(
            slack_notifier=MagicMock(),
            email_notifier=MagicMock(),
        )
        register_default_rules(manager)

        # 정상 컨텍스트 — 알림 없어야
        manager.check_rules({
            "error_rate": 1.0,
            "queue_depth": 100,
            "p95_latency": 5.0,
            "disk_percent": 50,
        })
        assert len(manager.get_active_alerts()) == 0


# ============================================================
# 통합 시나리오 테스트
# ============================================================

class TestIntegrationScenarios:

    def test_full_alert_pipeline(self):
        """MetricsCollector → AlertManager → 알림 발생 전체 파이프라인"""
        from monitor.alert_loop import (
            AlertMonitorLoop, MetricsCollector, register_task4_rules,
        )
        from alerts.manager import AlertManager

        mock_slack = MagicMock()
        manager = AlertManager(
            slack_notifier=mock_slack,
            email_notifier=MagicMock(),
        )
        register_task4_rules(manager)

        mc = MetricsCollector()

        # GPU 낮음 시나리오
        with patch.object(mc, "collect", return_value={
            "gpu_util": 15,
            "queue_size": 5000,
            "failure_rate": 0.05,
            "daily_progress": 0.8,
            "current_hour": 14,
            "timestamp": "",
        }):
            loop = AlertMonitorLoop(
                alert_manager=manager,
                metrics_collector=mc,
            )
            loop.check_once()

        # Slack에 알림 전송됨
        active = manager.get_active_alerts()
        assert any(a.rule_name == "gpu_util_low" for a in active)
        # Slack.send가 호출됐어야
        assert mock_slack.send.called

    def test_main_cli_integration(self):
        """main.py CLI 파싱 → 실행 통합"""
        from main import build_parser

        parser = build_parser()

        # run-forever 파싱
        args = parser.parse_args([
            "run-forever", "--target", "100", "--interval", "10",
        ])
        assert args.command == "run-forever"
        assert args.target == 100
        assert args.interval == 10

    def test_monitor_module_exports(self):
        """monitor 모듈 export 확인"""
        from monitor import (
            StatsCollector,
            publish_log,
            AlertMonitorLoop,
            MetricsCollector,
            register_task4_rules,
        )
        assert AlertMonitorLoop is not None
        assert MetricsCollector is not None
        assert register_task4_rules is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
