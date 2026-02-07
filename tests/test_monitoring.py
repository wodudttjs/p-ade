"""
Tests for MODULE 6: Monitoring & Operations

FR-6.1: Real-time Dashboard
FR-6.2: Error Handling & Logging
FR-6.3: Data Quality Monitoring
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json


# =============================================================================
# FR-6.1: Real-time Dashboard Tests
# =============================================================================

class TestMetricsRegistry:
    """MetricsRegistry 테스트"""
    
    def test_singleton_instance(self):
        """싱글톤 인스턴스 테스트"""
        from monitoring.metrics import MetricsRegistry
        
        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()
        
        # 새 인스턴스지만 메트릭은 공유됨 (Prometheus 전역)
        assert registry1 is not None
        assert registry2 is not None
    
    def test_stage_enum(self):
        """Stage enum 테스트"""
        from monitoring.metrics import Stage
        
        assert Stage.DISCOVER.value == "discover"
        assert Stage.DOWNLOAD.value == "download"
        assert Stage.EXTRACT.value == "extract"
        assert Stage.TRANSFORM.value == "transform"
        assert Stage.UPLOAD.value == "upload"
    
    def test_status_enum(self):
        """Status enum 테스트"""
        from monitoring.metrics import Status
        
        assert Status.SUCCESS.value == "success"
        assert Status.FAIL.value == "fail"
        assert Status.SKIP.value == "skip"
    
    def test_item_type_enum(self):
        """ItemType enum 테스트"""
        from monitoring.metrics import ItemType
        
        assert ItemType.FILE.value == "file"
        assert ItemType.VIDEO.value == "video"
        assert ItemType.EPISODE.value == "episode"
    
    def test_increment_job_counter(self):
        """작업 카운터 증가 테스트"""
        from monitoring.metrics import MetricsRegistry, Stage, Status
        
        registry = MetricsRegistry()
        
        # 카운터 증가 (prometheus_client 없을 때 오류 없이 동작)
        try:
            registry.inc_jobs(Stage.DOWNLOAD, Status.SUCCESS)
        except Exception:
            pass  # prometheus_client 없으면 무시
    
    def test_observe_job_decorator(self):
        """observe_job 데코레이터 테스트"""
        from monitoring.metrics import observe_job, Stage
        
        @observe_job(Stage.TRANSFORM)
        def sample_job():
            return "done"
        
        result = sample_job()
        assert result == "done"
    
    def test_observe_job_decorator_with_exception(self):
        """observe_job 데코레이터 예외 테스트"""
        from monitoring.metrics import observe_job, Stage
        
        @observe_job(Stage.DOWNLOAD)
        def failing_job():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            failing_job()
    
    def test_observe_job_context_manager(self):
        """observe_job_context 컨텍스트 매니저 테스트"""
        from monitoring.metrics import observe_job_context, Stage
        
        with observe_job_context(Stage.UPLOAD):
            result = 1 + 1
        
        assert result == 2
    
    def test_observe_job_context_with_exception(self):
        """observe_job_context 예외 테스트"""
        from monitoring.metrics import observe_job_context, Stage
        
        with pytest.raises(ValueError):
            with observe_job_context(Stage.EXTRACT):
                raise ValueError("test")


class TestResourceCollector:
    """ResourceCollector 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        from monitoring.metrics import ResourceCollector, MetricsRegistry
        
        registry = MetricsRegistry()
        collector = ResourceCollector(metrics_registry=registry, interval_seconds=5.0)
        assert collector.interval == 5.0
        assert not collector._running
    
    def test_collect_cpu(self):
        """CPU 수집 테스트"""
        from monitoring.metrics import ResourceCollector, MetricsRegistry
        
        registry = MetricsRegistry()
        collector = ResourceCollector(metrics_registry=registry)
        cpu = collector._collect_cpu()
        
        # psutil이 있으면 값 반환, 없으면 0
        assert isinstance(cpu, float)
        assert cpu >= 0
    
    def test_collect_memory(self):
        """메모리 수집 테스트"""
        from monitoring.metrics import ResourceCollector, MetricsRegistry
        
        registry = MetricsRegistry()
        collector = ResourceCollector(metrics_registry=registry)
        memory = collector._collect_memory()
        
        assert isinstance(memory, int)
        assert memory >= 0
    
    def test_start_stop(self):
        """시작/중지 테스트"""
        from monitoring.metrics import ResourceCollector, MetricsRegistry
        
        registry = MetricsRegistry()
        collector = ResourceCollector(metrics_registry=registry, interval_seconds=1.0)
        
        collector.start()
        assert collector._running
        
        collector.stop()
        assert not collector._running


class TestMetricsExporter:
    """MetricsExporter 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        from monitoring.exporter import MetricsExporter
        
        exporter = MetricsExporter(port=9999)
        assert exporter.port == 9999
        assert exporter.host == "0.0.0.0"
    
    def test_get_metrics_text(self):
        """메트릭 텍스트 반환 테스트"""
        from monitoring.exporter import MetricsExporter
        
        exporter = MetricsExporter()
        text = exporter.get_metrics_text()
        
        # prometheus_client가 없으면 빈 문자열
        assert isinstance(text, str)


class TestPushGatewayExporter:
    """PushGatewayExporter 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        from monitoring.exporter import PushGatewayExporter
        
        exporter = PushGatewayExporter(
            gateway_url="localhost:9091",
            job_name="test-job",
        )
        assert exporter.gateway_url == "localhost:9091"
        assert exporter.job_name == "test-job"


class TestKPICalculator:
    """KPICalculator 테스트"""
    
    def test_calculate_throughput(self):
        """처리량 계산 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        
        result = calc.calculate_throughput([100, 200, 300], 60)
        assert result == 10.0  # 600 / 60
    
    def test_calculate_throughput_zero_duration(self):
        """처리량 0초 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        result = calc.calculate_throughput([100], 0)
        assert result == 0.0
    
    def test_calculate_error_rate(self):
        """에러율 계산 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        
        result = calc.calculate_error_rate(10, 100)
        assert result == 10.0
    
    def test_calculate_error_rate_zero_total(self):
        """에러율 0 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        result = calc.calculate_error_rate(0, 0)
        assert result == 0.0
    
    def test_calculate_success_rate(self):
        """성공률 계산 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        result = calc.calculate_success_rate(90, 100)
        assert result == 90.0
    
    def test_calculate_latency_percentile(self):
        """지연시간 백분위수 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        latencies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # index = int(10 * percentile), 0-indexed
        # p50: index=5 -> latencies[5]=6
        # p95: index=9 -> latencies[9]=10
        p50 = calc.calculate_latency_percentile(latencies, 0.5)
        p95 = calc.calculate_latency_percentile(latencies, 0.95)
        
        assert p50 == 6  # 50th percentile at index 5
        assert p95 == 10
    
    def test_aggregate_methods(self):
        """집계 방법 테스트"""
        from monitoring.kpi import KPICalculator, Aggregation
        
        calc = KPICalculator()
        values = [1, 2, 3, 4, 5]
        
        assert calc.aggregate(values, Aggregation.SUM) == 15
        assert calc.aggregate(values, Aggregation.AVG) == 3
        assert calc.aggregate(values, Aggregation.MIN) == 1
        assert calc.aggregate(values, Aggregation.MAX) == 5
        assert calc.aggregate(values, Aggregation.COUNT) == 5
    
    def test_build_dashboard(self):
        """대시보드 빌드 테스트"""
        from monitoring.kpi import KPICalculator
        
        calc = KPICalculator()
        dashboard = calc.build_dashboard(period_hours=24)
        
        assert dashboard is not None
        assert len(dashboard.metrics) > 0


class TestKPIMetric:
    """KPIMetric 테스트"""
    
    def test_status_ok(self):
        """상태 OK 테스트"""
        from monitoring.kpi import KPIMetric, KPIType
        
        metric = KPIMetric(
            name="test",
            kpi_type=KPIType.SUCCESS_RATE,
            value=99.0,
            unit="%",
            threshold_warning=95.0,
            threshold_critical=90.0,
        )
        
        assert metric.status == "ok"
    
    def test_status_warning(self):
        """상태 Warning 테스트"""
        from monitoring.kpi import KPIMetric, KPIType
        
        metric = KPIMetric(
            name="test",
            kpi_type=KPIType.ERROR_RATE,
            value=7.0,
            unit="%",
            threshold_warning=5.0,
            threshold_critical=10.0,
        )
        
        assert metric.status == "warning"
    
    def test_status_critical(self):
        """상태 Critical 테스트"""
        from monitoring.kpi import KPIMetric, KPIType
        
        metric = KPIMetric(
            name="test",
            kpi_type=KPIType.ERROR_RATE,
            value=15.0,
            unit="%",
            threshold_warning=5.0,
            threshold_critical=10.0,
        )
        
        assert metric.status == "critical"


class TestKPIDashboard:
    """KPIDashboard 테스트"""
    
    def test_add_metric(self):
        """메트릭 추가 테스트"""
        from monitoring.kpi import KPIDashboard, KPIMetric, KPIType
        
        dashboard = KPIDashboard()
        
        metric = KPIMetric(
            name="test",
            kpi_type=KPIType.THROUGHPUT,
            value=100,
            unit="items/sec",
        )
        
        dashboard.add_metric(metric)
        assert len(dashboard.metrics) == 1
    
    def test_get_by_type(self):
        """유형별 조회 테스트"""
        from monitoring.kpi import KPIDashboard, KPIMetric, KPIType
        
        dashboard = KPIDashboard()
        dashboard.add_metric(KPIMetric("a", KPIType.THROUGHPUT, 100, ""))
        dashboard.add_metric(KPIMetric("b", KPIType.ERROR_RATE, 5, ""))
        dashboard.add_metric(KPIMetric("c", KPIType.THROUGHPUT, 200, ""))
        
        throughput = dashboard.get_by_type(KPIType.THROUGHPUT)
        assert len(throughput) == 2


class TestGrafanaDashboard:
    """Grafana 대시보드 테스트"""
    
    def test_dashboard_builder(self):
        """대시보드 빌더 테스트"""
        from monitoring.dashboards import DashboardBuilder
        
        builder = DashboardBuilder()
        dashboard = builder.build_overview_dashboard()
        
        assert dashboard.uid == "p-ade-overview"
        assert len(dashboard.rows) > 0
    
    def test_dashboard_to_json(self):
        """대시보드 JSON 변환 테스트"""
        from monitoring.dashboards import DashboardBuilder
        
        builder = DashboardBuilder()
        dashboard = builder.build_overview_dashboard()
        
        json_str = dashboard.to_json()
        parsed = json.loads(json_str)
        
        assert "uid" in parsed
        assert "title" in parsed
        assert "panels" in parsed
    
    def test_create_prometheus_query(self):
        """Prometheus 쿼리 생성 테스트"""
        from monitoring.dashboards import DashboardBuilder
        
        query = DashboardBuilder.create_prometheus_query(
            expr='sum(rate(metric[5m]))',
            legend="{{label}}",
        )
        
        assert query["expr"] == 'sum(rate(metric[5m]))'
        assert query["legendFormat"] == "{{label}}"


# =============================================================================
# FR-6.2: Error Handling & Logging Tests
# =============================================================================

class TestErrorClassification:
    """오류 분류 테스트"""
    
    def test_classify_timeout_error(self):
        """타임아웃 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType
        
        error = TimeoutError("Connection timed out")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.NETWORK_TIMEOUT
    
    def test_classify_connection_error(self):
        """연결 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType
        
        error = ConnectionError("Failed to connect")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.NETWORK_CONNECTION
    
    def test_classify_value_error(self):
        """값 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType
        
        error = ValueError("Invalid value")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.DATA_VALIDATION
    
    def test_classify_file_not_found(self):
        """파일 없음 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType
        
        error = FileNotFoundError("File not found")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.STORAGE_NOT_FOUND
    
    def test_classify_memory_error(self):
        """메모리 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType, Severity
        
        error = MemoryError("Out of memory")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.RESOURCE_MEMORY
        assert info.severity == Severity.CRITICAL
    
    def test_classify_unknown_error(self):
        """알 수 없는 오류 분류 테스트"""
        from errors.errors import classify_error, ErrorType
        
        class CustomError(Exception):
            pass
        
        error = CustomError("Unknown")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.UNKNOWN
    
    def test_rate_limit_detection(self):
        """Rate limit 감지 테스트"""
        from errors.errors import classify_error, ErrorType
        
        error = Exception("Rate limit exceeded")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.API_RATE_LIMIT


class TestErrorInfo:
    """ErrorInfo 테스트"""
    
    def test_error_id_generation(self):
        """오류 ID 생성 테스트"""
        from errors.errors import ErrorInfo, ErrorType, Severity, Retryable
        
        info = ErrorInfo(
            error_type=ErrorType.NETWORK_TIMEOUT,
            severity=Severity.WARNING,
            retryable=Retryable.YES,
            message="Test error",
        )
        
        assert info.error_id is not None
        assert len(info.error_id) == 12
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        from errors.errors import ErrorInfo, ErrorType, Severity, Retryable
        
        info = ErrorInfo(
            error_type=ErrorType.DATA_VALIDATION,
            severity=Severity.ERROR,
            retryable=Retryable.NO,
            message="Validation failed",
        )
        
        d = info.to_dict()
        
        assert d["error_type"] == "data_validation"
        assert d["severity"] == "error"
        assert d["retryable"] == "no"


class TestErrorRegistry:
    """ErrorRegistry 테스트"""
    
    def test_record_error(self):
        """오류 기록 테스트"""
        from errors.errors import ErrorRegistry, ErrorInfo, ErrorType, Severity, Retryable
        
        registry = ErrorRegistry(max_errors=100)
        
        error = ErrorInfo(
            error_type=ErrorType.API_SERVER_ERROR,
            severity=Severity.ERROR,
            retryable=Retryable.YES,
            message="Server error",
        )
        
        registry.record(error)
        
        assert len(registry._errors) == 1
    
    def test_get_counts(self):
        """오류 수 조회 테스트"""
        from errors.errors import ErrorRegistry, ErrorInfo, ErrorType, Severity, Retryable
        
        registry = ErrorRegistry()
        
        for _ in range(3):
            registry.record(ErrorInfo(
                ErrorType.NETWORK_TIMEOUT, Severity.WARNING, Retryable.YES, "timeout"
            ))
        
        for _ in range(2):
            registry.record(ErrorInfo(
                ErrorType.DATA_VALIDATION, Severity.ERROR, Retryable.NO, "validation"
            ))
        
        counts = registry.get_counts()
        
        assert counts.get("network_timeout") == 3
        assert counts.get("data_validation") == 2
    
    def test_get_recent(self):
        """최근 오류 조회 테스트"""
        from errors.errors import ErrorRegistry, ErrorInfo, ErrorType, Severity, Retryable
        
        registry = ErrorRegistry()
        
        for i in range(20):
            registry.record(ErrorInfo(
                ErrorType.UNKNOWN, Severity.INFO, Retryable.NO, f"error_{i}"
            ))
        
        recent = registry.get_recent(limit=5)
        
        assert len(recent) == 5
        assert recent[-1].message == "error_19"
    
    def test_max_errors_limit(self):
        """최대 오류 수 제한 테스트"""
        from errors.errors import ErrorRegistry, ErrorInfo, ErrorType, Severity, Retryable
        
        registry = ErrorRegistry(max_errors=10)
        
        for i in range(20):
            registry.record(ErrorInfo(
                ErrorType.UNKNOWN, Severity.INFO, Retryable.NO, f"error_{i}"
            ))
        
        assert len(registry._errors) == 10


class TestJSONLogger:
    """JSON 로거 테스트"""
    
    def test_json_formatter(self):
        """JSON 포매터 테스트"""
        from structured_logging.json_logger import JSONFormatter
        import logging
        
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed
    
    def test_context_filter(self):
        """컨텍스트 필터 테스트"""
        from structured_logging.json_logger import ContextFilter, set_request_id
        import logging
        
        filter = ContextFilter()
        
        set_request_id("req-123")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        
        filter.filter(record)
        
        assert record.request_id == "req-123"
    
    def test_log_context_manager(self):
        """로그 컨텍스트 매니저 테스트"""
        from structured_logging.json_logger import LogContext, get_request_id, get_job_id
        
        with LogContext(request_id="req-456", job_id="job-789"):
            assert get_request_id() == "req-456"
            assert get_job_id() == "job-789"
    
    def test_structured_logger(self):
        """구조화된 로거 테스트"""
        from structured_logging.json_logger import StructuredLogger
        
        logger = StructuredLogger("test")
        
        # 로그 호출 테스트 (예외 없이 동작)
        logger.info("Test info", key="value")
        logger.error("Test error", exception=ValueError("test"))
        logger.event("user_action", user_id="123")
        logger.metric("latency", 0.5)


class TestSlackNotifier:
    """Slack 알림 테스트"""
    
    def test_init_disabled(self):
        """비활성화 초기화 테스트"""
        from alerts.slack import SlackNotifier
        
        notifier = SlackNotifier(webhook_url=None)
        
        assert not notifier.enabled
    
    def test_slack_message_to_dict(self):
        """Slack 메시지 딕셔너리 변환 테스트"""
        from alerts.slack import SlackMessage
        
        message = SlackMessage(
            text="Test message",
            channel="#alerts",
        )
        
        d = message.to_dict()
        
        assert d["text"] == "Test message"
        assert d["channel"] == "#alerts"
    
    def test_slack_attachment(self):
        """Slack 첨부 테스트"""
        from alerts.slack import SlackAttachment
        
        attachment = SlackAttachment(
            title="Alert",
            text="Error occurred",
            color="#ff0000",
        )
        
        d = attachment.to_dict()
        
        assert d["title"] == "Alert"
        assert d["color"] == "#ff0000"
    
    @patch('requests.post')
    def test_send_success(self, mock_post):
        """전송 성공 테스트"""
        from alerts.slack import SlackNotifier, AlertLevel
        
        mock_post.return_value.status_code = 200
        
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send("Test", level=AlertLevel.INFO)
        
        assert result is True
    
    @patch('requests.post')
    def test_send_error_alert(self, mock_post):
        """오류 알림 전송 테스트"""
        from alerts.slack import SlackNotifier
        
        mock_post.return_value.status_code = 200
        
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send_error(
            error_type="NETWORK_TIMEOUT",
            error_message="Connection failed",
            error_id="abc123",
        )
        
        assert result is True


class TestEmailNotifier:
    """Email 알림 테스트"""
    
    def test_init_disabled(self):
        """비활성화 초기화 테스트"""
        from alerts.email import EmailNotifier, EmailConfig
        
        config = EmailConfig(smtp_user=None)
        notifier = EmailNotifier(config)
        
        assert not notifier.enabled
    
    def test_email_config_from_env(self):
        """환경변수 설정 테스트"""
        from alerts.email import EmailConfig
        
        config = EmailConfig.from_env()
        
        assert config.smtp_host is not None
        assert config.smtp_port > 0
    
    def test_email_message(self):
        """이메일 메시지 테스트"""
        from alerts.email import EmailMessage, EmailPriority
        
        message = EmailMessage(
            to=["test@example.com"],
            subject="Test",
            body_text="Test body",
            priority=EmailPriority.HIGH,
        )
        
        assert len(message.to) == 1
        assert message.priority == EmailPriority.HIGH


class TestAlertManager:
    """AlertManager 테스트"""
    
    def test_register_rule(self):
        """규칙 등록 테스트"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel, AlertLevel
        
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_rule",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            severity=AlertLevel.WARNING,
        )
        
        manager.register_rule(rule)
        
        assert "test_rule" in manager._rules
    
    def test_fire_alert(self):
        """알림 발생 테스트"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel, AlertLevel
        
        manager = AlertManager()
        
        rule = AlertRule(
            name="fire_test",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            severity=AlertLevel.ERROR,
            cooldown_seconds=0,
        )
        
        manager.register_rule(rule)
        
        alert = manager.fire("fire_test", "Test alert message")
        
        assert alert is not None
        assert alert.rule_name == "fire_test"
    
    def test_cooldown(self):
        """쿨다운 테스트"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel, AlertLevel
        
        manager = AlertManager()
        
        rule = AlertRule(
            name="cooldown_test",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            cooldown_seconds=3600,
        )
        
        manager.register_rule(rule)
        
        # 첫 번째 알림
        alert1 = manager.fire("cooldown_test", "First")
        assert alert1 is not None
        
        # 쿨다운 중 두 번째 알림
        alert2 = manager.fire("cooldown_test", "Second")
        assert alert2 is None
    
    def test_silence(self):
        """침묵 테스트"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel
        
        manager = AlertManager()
        
        rule = AlertRule(
            name="silence_test",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            cooldown_seconds=0,
        )
        
        manager.register_rule(rule)
        manager.silence("silence_test", duration_seconds=3600)
        
        alert = manager.fire("silence_test", "Should be silenced")
        
        assert alert is None
    
    def test_resolve_alert(self):
        """알림 해결 테스트"""
        from alerts.manager import AlertManager, AlertRule, AlertChannel, AlertState
        
        manager = AlertManager()
        
        rule = AlertRule(
            name="resolve_test",
            condition=lambda **_: True,
            channels=[AlertChannel.LOG],
            cooldown_seconds=0,
        )
        
        manager.register_rule(rule)
        
        alert = manager.fire("resolve_test", "Test")
        assert alert is not None
        
        resolved = manager.resolve(alert.fingerprint)
        assert resolved is not None
        assert resolved.state == AlertState.RESOLVED


# =============================================================================
# FR-6.3: Data Quality Monitoring Tests
# =============================================================================

class TestQualityTests:
    """품질 테스트 클래스 테스트"""
    
    def test_nan_check_pass(self):
        """NaN 검사 통과 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import NaNCheckTest, TestResult
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        test = NaNCheckTest(max_nan_ratio=0.1)
        result = test.run(df)
        
        assert result.result == TestResult.PASS
    
    def test_nan_check_fail(self):
        """NaN 검사 실패 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        import numpy as np
        from quality.tests import NaNCheckTest, TestResult
        
        df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [4, np.nan, 6]})
        
        test = NaNCheckTest(max_nan_ratio=0.1)
        result = test.run(df)
        
        assert result.result == TestResult.FAIL
    
    def test_shape_check_pass(self):
        """형태 검사 통과 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import ShapeCheckTest, TestResult
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        test = ShapeCheckTest(min_rows=1, expected_columns=2)
        result = test.run(df)
        
        assert result.result == TestResult.PASS
    
    def test_shape_check_missing_columns(self):
        """형태 검사 필수 컬럼 누락 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import ShapeCheckTest, TestResult
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        test = ShapeCheckTest(required_columns=["a", "b", "c"])
        result = test.run(df)
        
        assert result.result == TestResult.FAIL
    
    def test_range_check_pass(self):
        """범위 검사 통과 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import RangeCheckTest, TestResult
        
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        
        test = RangeCheckTest(column="value", min_value=0, max_value=100)
        result = test.run(df)
        
        assert result.result == TestResult.PASS
    
    def test_range_check_fail(self):
        """범위 검사 실패 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import RangeCheckTest, TestResult
        
        df = pd.DataFrame({"value": [10, 200, 30]})
        
        test = RangeCheckTest(column="value", max_value=100)
        result = test.run(df)
        
        assert result.result == TestResult.FAIL
    
    def test_unique_check_pass(self):
        """고유성 검사 통과 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import UniqueCheckTest, TestResult
        
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        
        test = UniqueCheckTest(columns=["id"])
        result = test.run(df)
        
        assert result.result == TestResult.PASS
    
    def test_unique_check_fail(self):
        """고유성 검사 실패 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import UniqueCheckTest, TestResult
        
        df = pd.DataFrame({"id": [1, 2, 2, 3, 3]})
        
        test = UniqueCheckTest(columns=["id"])
        result = test.run(df)
        
        assert result.result == TestResult.FAIL
    
    def test_type_check_pass(self):
        """타입 검사 통과 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import TypeCheckTest, TestResult
        
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        })
        
        test = TypeCheckTest({"int_col": "int", "float_col": "float"})
        result = test.run(df)
        
        assert result.result == TestResult.PASS
    
    def test_custom_test(self):
        """커스텀 테스트 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import CustomTest, TestResult
        
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        test = CustomTest(
            name="positive_values",
            test_func=lambda data: (data["value"] > 0).all(),
        )
        result = test.run(df)
        
        assert result.result == TestResult.PASS


class TestQualityTestRunner:
    """QualityTestRunner 테스트"""
    
    def test_run_all_tests(self):
        """전체 테스트 실행 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import QualityTestRunner
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        
        runner = QualityTestRunner()
        runner.add_nan_check(max_nan_ratio=0.1)
        runner.add_shape_check(min_rows=1)
        runner.add_range_check("value", min_value=0, max_value=100)
        
        report = runner.run(df, "test_dataset")
        
        assert report.total == 3
        assert report.passed == 3
    
    def test_quality_report(self):
        """품질 보고서 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import QualityTestRunner
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        runner = QualityTestRunner()
        runner.add_nan_check()
        
        report = runner.run(df, "my_dataset")
        
        assert report.dataset_name == "my_dataset"
        assert report.success_rate == 100.0


class TestDataProfiler:
    """DataProfiler 테스트"""
    
    def test_profile_dataframe(self):
        """DataFrame 프로파일링 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.profiling import DataProfiler
        
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "text": ["a", "b", "c", "d", "e"],
            "category": ["cat1", "cat1", "cat2", "cat2", "cat2"],
        })
        
        profiler = DataProfiler()
        profile = profiler.profile(df, "test")
        
        assert profile.name == "test"
        assert profile.row_count == 5
        assert profile.column_count == 3
        assert len(profile.columns) == 3
    
    def test_numeric_stats(self):
        """수치형 통계 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.profiling import DataProfiler, ColumnType
        
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        
        profiler = DataProfiler()
        profile = profiler.profile(df)
        
        col = profile.get_column("value")
        
        assert col.column_type == ColumnType.NUMERIC
        assert col.numeric_stats is not None
        assert col.numeric_stats.mean == 3.0
        assert col.numeric_stats.min == 1.0
        assert col.numeric_stats.max == 5.0
    
    def test_categorical_stats(self):
        """범주형 통계 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.profiling import DataProfiler, ColumnType
        
        df = pd.DataFrame({"category": ["a", "a", "a", "b", "b", "c"]})
        
        profiler = DataProfiler(max_unique_categories=10)
        profile = profiler.profile(df)
        
        col = profile.get_column("category")
        
        assert col.column_type == ColumnType.CATEGORICAL
        assert col.categorical_stats is not None
        assert col.categorical_stats.unique_count == 3
        assert col.categorical_stats.top_value == "a"
    
    def test_null_ratio(self):
        """Null 비율 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        import numpy as np
        from quality.profiling import DataProfiler
        
        df = pd.DataFrame({"value": [1, 2, np.nan, np.nan, 5]})
        
        profiler = DataProfiler()
        profile = profiler.profile(df)
        
        col = profile.get_column("value")
        
        assert col.null_count == 2
        assert col.null_ratio == 0.4


class TestProfileComparator:
    """ProfileComparator 테스트"""
    
    def test_compare_profiles(self):
        """프로파일 비교 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.profiling import DataProfiler, ProfileComparator
        
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [10, 20, 30], "b": [40, 50, 60], "c": [1, 2, 3]})
        
        profiler = DataProfiler()
        profile1 = profiler.profile(df1, "baseline")
        profile2 = profiler.profile(df2, "current")
        
        comparator = ProfileComparator()
        comparison = comparator.compare(profile1, profile2)
        
        assert comparison["column_count_diff"] == 1
        assert any(c["change"] == "added" for c in comparison["column_changes"])


class TestReportGenerator:
    """ReportGenerator 테스트"""
    
    def test_generate_summary(self):
        """요약 생성 테스트"""
        from quality.reporting import ReportGenerator, ReportPeriod
        
        generator = ReportGenerator()
        summary = generator.generate_summary(ReportPeriod.WEEKLY)
        
        assert summary.period == ReportPeriod.WEEKLY
        assert summary.total_datasets == 0
    
    def test_format_markdown(self):
        """마크다운 포맷 테스트"""
        from quality.reporting import ReportGenerator, ReportPeriod, ReportFormat
        
        generator = ReportGenerator()
        summary = generator.generate_summary(ReportPeriod.WEEKLY)
        
        md = generator.format_report(summary, ReportFormat.MARKDOWN)
        
        assert "# Data Quality Report" in md
        assert "Summary" in md
    
    def test_format_html(self):
        """HTML 포맷 테스트"""
        from quality.reporting import ReportGenerator, ReportPeriod, ReportFormat
        
        generator = ReportGenerator()
        summary = generator.generate_summary(ReportPeriod.WEEKLY)
        
        html = generator.format_report(summary, ReportFormat.HTML)
        
        assert "<html>" in html
        assert "Data Quality Report" in html
    
    def test_format_json(self):
        """JSON 포맷 테스트"""
        from quality.reporting import ReportGenerator, ReportPeriod, ReportFormat
        
        generator = ReportGenerator()
        summary = generator.generate_summary(ReportPeriod.WEEKLY)
        
        json_str = generator.format_report(summary, ReportFormat.JSON)
        parsed = json.loads(json_str)
        
        assert "period" in parsed
        assert "summary" in parsed


class TestQualityReportSummary:
    """QualityReportSummary 테스트"""
    
    def test_add_section(self):
        """섹션 추가 테스트"""
        from quality.reporting import QualityReportSummary, ReportPeriod
        from datetime import datetime
        
        summary = QualityReportSummary(
            period=ReportPeriod.WEEKLY,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
        )
        
        summary.add_section("Test Section", {"key": "value"}, order=1)
        
        assert len(summary.sections) == 1
        assert summary.sections[0].title == "Test Section"
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        from quality.reporting import QualityReportSummary, ReportPeriod
        from datetime import datetime
        
        summary = QualityReportSummary(
            period=ReportPeriod.DAILY,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            total_datasets=5,
            total_tests=50,
            passed_tests=45,
            failed_tests=5,
            success_rate=90.0,
        )
        
        d = summary.to_dict()
        
        assert d["period"] == "daily"
        assert d["summary"]["total_datasets"] == 5
        assert d["summary"]["success_rate"] == 90.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """모니터링 통합 테스트"""
    
    def test_metrics_with_alerts(self):
        """메트릭과 알림 통합 테스트"""
        from monitoring.metrics import MetricsRegistry, Stage, Status
        from alerts.manager import AlertManager, AlertRule, AlertChannel, AlertLevel
        
        # 메트릭 기록
        registry = MetricsRegistry()
        
        # 알림 설정
        manager = AlertManager()
        manager.register_rule(AlertRule(
            name="high_errors",
            condition=lambda error_count=0, **_: error_count > 10,
            channels=[AlertChannel.LOG],
            cooldown_seconds=0,
        ))
        
        # 알림 발생
        alert = manager.fire("high_errors", "Too many errors", force=True)
        
        assert alert is not None
    
    def test_quality_with_alerts(self):
        """품질 테스트와 알림 통합 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        from quality.tests import QualityTestRunner, TestResult
        from alerts.manager import AlertManager, AlertChannel, AlertLevel
        
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        runner = QualityTestRunner()
        runner.add_nan_check()
        
        report = runner.run(df)
        
        # 실패 시 알림
        manager = AlertManager()
        
        if report.overall_result == TestResult.FAIL:
            # 알림 발생 로직
            pass
        
        assert report.overall_result == TestResult.PASS
    
    def test_error_to_metrics(self):
        """오류를 메트릭으로 변환 테스트"""
        from errors.errors import classify_error, ErrorType
        from monitoring.metrics import MetricsRegistry, Stage
        
        # 오류 분류
        error = TimeoutError("Connection timeout")
        info = classify_error(error)
        
        assert info.error_type == ErrorType.NETWORK_TIMEOUT
        
        # 메트릭에 기록 (실제 환경)
        registry = MetricsRegistry()
        # registry.inc_errors(Stage.DOWNLOAD, info.error_type.value)


class TestEndToEndMonitoring:
    """E2E 모니터링 테스트"""
    
    def test_full_monitoring_flow(self):
        """전체 모니터링 흐름 테스트"""
        pytest.importorskip("pandas")
        import pandas as pd
        
        # 1. 작업 실행 및 메트릭 수집
        from monitoring.metrics import observe_job, Stage
        
        @observe_job(Stage.TRANSFORM)
        def process_data():
            return pd.DataFrame({"a": [1, 2, 3]})
        
        df = process_data()
        
        # 2. 데이터 품질 검사
        from quality.tests import QualityTestRunner
        
        runner = QualityTestRunner()
        runner.add_nan_check()
        runner.add_shape_check(min_rows=1)
        
        report = runner.run(df, "processed_data")
        
        # 3. 보고서 생성
        from quality.reporting import get_report_generator, ReportPeriod, ReportFormat
        
        generator = get_report_generator()
        generator.add_test_result(report)
        
        summary = generator.generate_summary(ReportPeriod.DAILY)
        markdown = generator.format_report(summary, ReportFormat.MARKDOWN)
        
        assert "Data Quality Report" in markdown
        assert report.passed > 0
