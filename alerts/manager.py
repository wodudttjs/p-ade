"""
Alert Manager

통합 알림 관리 시스템
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import threading

from core.logging_config import setup_logger
from alerts.slack import SlackNotifier, AlertLevel, get_slack_notifier
from alerts.email import EmailNotifier, EmailPriority, get_email_notifier

logger = setup_logger(__name__)


class AlertChannel(str, Enum):
    """알림 채널"""
    SLACK = "slack"
    EMAIL = "email"
    LOG = "log"
    WEBHOOK = "webhook"


class AlertState(str, Enum):
    """알림 상태"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    condition: Callable[..., bool]
    channels: List[AlertChannel]
    severity: AlertLevel = AlertLevel.WARNING
    cooldown_seconds: int = 300  # 5분 쿨다운
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, AlertRule):
            return self.name == other.name
        return False


@dataclass
class Alert:
    """알림 인스턴스"""
    rule_name: str
    state: AlertState
    severity: AlertLevel
    message: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """알림 핑거프린트 생성"""
        content = f"{self.rule_name}:{sorted(self.labels.items())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "state": self.state.value,
            "severity": self.severity.value,
            "message": self.message,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "labels": self.labels,
            "annotations": self.annotations,
            "fingerprint": self.fingerprint,
        }


class AlertManager:
    """
    통합 알림 관리자
    
    Task 6.2.5: 알림 통합 관리
    """
    
    def __init__(
        self,
        slack_notifier: Optional[SlackNotifier] = None,
        email_notifier: Optional[EmailNotifier] = None,
        email_recipients: Optional[List[str]] = None,
    ):
        self.slack = slack_notifier or get_slack_notifier()
        self.email = email_notifier or get_email_notifier()
        self.email_recipients = email_recipients or []
        
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._last_fired: Dict[str, datetime] = {}
        self._silenced: Set[str] = set()
        self._history: List[Alert] = []
        self._max_history = 1000
        self._lock = threading.Lock()
    
    def register_rule(self, rule: AlertRule):
        """알림 규칙 등록"""
        with self._lock:
            self._rules[rule.name] = rule
            logger.info(f"Registered alert rule: {rule.name}")
    
    def unregister_rule(self, rule_name: str):
        """알림 규칙 해제"""
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                logger.info(f"Unregistered alert rule: {rule_name}")
    
    def silence(self, rule_name: str, duration_seconds: int = 3600):
        """알림 일시 중지"""
        with self._lock:
            self._silenced.add(rule_name)
            logger.info(f"Silenced alert: {rule_name} for {duration_seconds}s")
            
            # 자동 해제 스케줄링
            def unsilence():
                import time
                time.sleep(duration_seconds)
                with self._lock:
                    self._silenced.discard(rule_name)
                    logger.info(f"Unsilenced alert: {rule_name}")
            
            thread = threading.Thread(target=unsilence, daemon=True)
            thread.start()
    
    def unsilence(self, rule_name: str):
        """알림 재개"""
        with self._lock:
            self._silenced.discard(rule_name)
    
    def fire(
        self,
        rule_name: str,
        message: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        force: bool = False,
    ) -> Optional[Alert]:
        """
        알림 발생
        
        Args:
            rule_name: 규칙 이름
            message: 알림 메시지
            labels: 레이블
            annotations: 주석
            force: 쿨다운 무시
        
        Returns:
            생성된 알림 또는 None (쿨다운/침묵 중)
        """
        with self._lock:
            # 규칙 확인
            rule = self._rules.get(rule_name)
            if not rule:
                logger.warning(f"Unknown alert rule: {rule_name}")
                return None
            
            # 침묵 확인
            if rule_name in self._silenced:
                logger.debug(f"Alert silenced: {rule_name}")
                return None
            
            # 쿨다운 확인
            last = self._last_fired.get(rule_name)
            if not force and last:
                elapsed = (datetime.utcnow() - last).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    logger.debug(f"Alert in cooldown: {rule_name}")
                    return None
            
            # 알림 생성
            alert = Alert(
                rule_name=rule_name,
                state=AlertState.FIRING,
                severity=rule.severity,
                message=message,
                fired_at=datetime.utcnow(),
                labels={**rule.labels, **(labels or {})},
                annotations={**rule.annotations, **(annotations or {})},
            )
            
            # 활성 알림에 추가
            self._active_alerts[alert.fingerprint] = alert
            self._last_fired[rule_name] = datetime.utcnow()
            
            # 히스토리에 추가
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history.pop(0)
        
        # 알림 전송 (락 밖에서)
        self._notify(alert, rule.channels)
        
        return alert
    
    def resolve(self, fingerprint: str) -> Optional[Alert]:
        """알림 해결"""
        with self._lock:
            if fingerprint not in self._active_alerts:
                return None
            
            alert = self._active_alerts[fingerprint]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            del self._active_alerts[fingerprint]
            
            logger.info(f"Alert resolved: {alert.rule_name}")
            return alert
    
    def _notify(self, alert: Alert, channels: List[AlertChannel]):
        """채널별 알림 전송"""
        for channel in channels:
            try:
                if channel == AlertChannel.SLACK:
                    self._notify_slack(alert)
                elif channel == AlertChannel.EMAIL:
                    self._notify_email(alert)
                elif channel == AlertChannel.LOG:
                    self._notify_log(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._notify_webhook(alert)
            except Exception as e:
                logger.error(f"Failed to notify {channel.value}: {e}")
    
    def _notify_slack(self, alert: Alert):
        """Slack 알림"""
        self.slack.send(
            text=alert.message,
            level=alert.severity,
            title=f"Alert: {alert.rule_name}",
            fields=alert.labels,
        )
    
    def _notify_email(self, alert: Alert):
        """이메일 알림"""
        if not self.email_recipients:
            return
        
        priority = EmailPriority.HIGH
        if alert.severity == AlertLevel.CRITICAL:
            priority = EmailPriority.HIGH
        elif alert.severity == AlertLevel.INFO:
            priority = EmailPriority.NORMAL
        
        self.email.send_alert(
            to=self.email_recipients,
            alert_type=f"{alert.severity.value.upper()} - {alert.rule_name}",
            alert_message=alert.message,
            details=alert.labels,
            priority=priority,
        )
    
    def _notify_log(self, alert: Alert):
        """로그 알림"""
        log_method = getattr(logger, alert.severity.value, logger.warning)
        log_method(
            f"ALERT [{alert.rule_name}]: {alert.message}",
            extra={"alert": alert.to_dict()},
        )
    
    def _notify_webhook(self, alert: Alert):
        """웹훅 알림 (구현 필요)"""
        pass
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_history(
        self,
        limit: int = 100,
        severity: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """알림 히스토리 조회"""
        with self._lock:
            alerts = self._history[-limit:]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts
    
    def check_rules(self, context: Dict[str, Any]):
        """등록된 규칙 평가"""
        with self._lock:
            rules = list(self._rules.values())
        
        for rule in rules:
            try:
                if rule.condition(**context):
                    self.fire(
                        rule_name=rule.name,
                        message=rule.description or f"Alert: {rule.name}",
                    )
            except Exception as e:
                logger.error(f"Failed to check rule {rule.name}: {e}")


# 싱글톤 인스턴스
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """알림 관리자 싱글톤 반환"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# 기본 알림 규칙 등록
def register_default_rules(manager: AlertManager):
    """기본 알림 규칙 등록"""
    
    # 에러율 규칙
    manager.register_rule(AlertRule(
        name="high_error_rate",
        condition=lambda error_rate=0, **_: error_rate > 10.0,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        severity=AlertLevel.ERROR,
        cooldown_seconds=600,
        description="Error rate exceeded 10%",
    ))
    
    # 큐 깊이 규칙
    manager.register_rule(AlertRule(
        name="high_queue_depth",
        condition=lambda queue_depth=0, **_: queue_depth > 5000,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        severity=AlertLevel.WARNING,
        cooldown_seconds=300,
        description="Queue depth exceeded 5000",
    ))
    
    # 지연 시간 규칙
    manager.register_rule(AlertRule(
        name="high_latency",
        condition=lambda p95_latency=0, **_: p95_latency > 60.0,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        severity=AlertLevel.WARNING,
        cooldown_seconds=300,
        description="P95 latency exceeded 60 seconds",
    ))
    
    # 디스크 사용량 규칙
    manager.register_rule(AlertRule(
        name="disk_usage_high",
        condition=lambda disk_percent=0, **_: disk_percent > 90,
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.LOG],
        severity=AlertLevel.CRITICAL,
        cooldown_seconds=1800,
        description="Disk usage exceeded 90%",
    ))
