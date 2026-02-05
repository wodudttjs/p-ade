"""
Slack Alert Notifier

Slack 웹훅을 통한 알림 전송
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import os

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class AlertLevel(str, Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Slack 색상 매핑
LEVEL_COLORS = {
    AlertLevel.INFO: "#36a64f",  # green
    AlertLevel.WARNING: "#ffcc00",  # yellow
    AlertLevel.ERROR: "#ff6600",  # orange
    AlertLevel.CRITICAL: "#ff0000",  # red
}

LEVEL_EMOJIS = {
    AlertLevel.INFO: ":information_source:",
    AlertLevel.WARNING: ":warning:",
    AlertLevel.ERROR: ":x:",
    AlertLevel.CRITICAL: ":rotating_light:",
}


@dataclass
class SlackAttachment:
    """Slack 메시지 첨부"""
    title: str
    text: str
    color: str = "#36a64f"
    fields: List[Dict[str, Any]] = field(default_factory=list)
    footer: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        attachment = {
            "title": self.title,
            "text": self.text,
            "color": self.color,
            "fields": self.fields,
        }
        if self.footer:
            attachment["footer"] = self.footer
        if self.timestamp:
            attachment["ts"] = int(self.timestamp.timestamp())
        return attachment


@dataclass
class SlackMessage:
    """Slack 메시지"""
    text: str
    channel: Optional[str] = None
    username: str = "P-ADE Alert Bot"
    icon_emoji: str = ":robot_face:"
    attachments: List[SlackAttachment] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        message = {
            "text": self.text,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
        }
        if self.channel:
            message["channel"] = self.channel
        if self.attachments:
            message["attachments"] = [a.to_dict() for a in self.attachments]
        if self.blocks:
            message["blocks"] = self.blocks
        return message


class SlackNotifier:
    """
    Slack 알림 전송기
    
    Task 6.2.3: Slack 알림 시스템
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        default_channel: Optional[str] = None,
        enabled: bool = True,
    ):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.default_channel = default_channel
        self.enabled = enabled and self.webhook_url is not None
        
        if not self.enabled:
            logger.warning("Slack notifications disabled (no webhook URL)")
    
    def _send_request(self, message: SlackMessage) -> bool:
        """웹훅 요청 전송"""
        if not self.enabled:
            logger.debug(f"Slack disabled, skipping: {message.text}")
            return False
        
        try:
            import requests
            
            response = requests.post(
                self.webhook_url,
                json=message.to_dict(),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            
            if response.status_code == 200:
                logger.debug(f"Slack message sent: {message.text[:50]}...")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False
                
        except ImportError:
            logger.warning("requests library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def send(
        self,
        text: str,
        level: AlertLevel = AlertLevel.INFO,
        title: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """
        알림 전송
        
        Args:
            text: 메시지 텍스트
            level: 알림 레벨
            title: 첨부 제목
            fields: 추가 필드
            channel: 채널 (없으면 기본 채널 사용)
        
        Returns:
            성공 여부
        """
        emoji = LEVEL_EMOJIS.get(level, ":bell:")
        
        attachments = []
        if title or fields:
            attachment_fields = []
            if fields:
                attachment_fields = [
                    {"title": k, "value": v, "short": len(str(v)) < 30}
                    for k, v in fields.items()
                ]
            
            attachments.append(SlackAttachment(
                title=title or level.value.upper(),
                text=text,
                color=LEVEL_COLORS.get(level, "#36a64f"),
                fields=attachment_fields,
                footer="P-ADE Pipeline",
                timestamp=datetime.utcnow(),
            ))
            message_text = f"{emoji} {level.value.upper()}"
        else:
            message_text = f"{emoji} {text}"
        
        message = SlackMessage(
            text=message_text,
            channel=channel or self.default_channel,
            attachments=attachments,
        )
        
        return self._send_request(message)
    
    def send_error(
        self,
        error_type: str,
        error_message: str,
        error_id: Optional[str] = None,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """오류 알림 전송"""
        fields = {"Error Type": error_type}
        if error_id:
            fields["Error ID"] = error_id
        if stage:
            fields["Stage"] = stage
        if context:
            for k, v in context.items():
                fields[k] = str(v)[:100]
        
        return self.send(
            text=error_message[:500],
            level=AlertLevel.ERROR,
            title="Pipeline Error",
            fields=fields,
        )
    
    def send_critical(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """크리티컬 알림 전송"""
        fields = details or {}
        return self.send(
            text=message,
            level=AlertLevel.CRITICAL,
            title="CRITICAL ALERT",
            fields={k: str(v) for k, v in fields.items()},
        )
    
    def send_job_complete(
        self,
        job_id: str,
        stage: str,
        duration_seconds: float,
        items_processed: int,
    ) -> bool:
        """작업 완료 알림"""
        return self.send(
            text=f"Job `{job_id}` completed",
            level=AlertLevel.INFO,
            title="Job Complete",
            fields={
                "Job ID": job_id,
                "Stage": stage,
                "Duration": f"{duration_seconds:.2f}s",
                "Items": str(items_processed),
            },
        )
    
    def send_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
    ) -> bool:
        """임계값 초과 알림"""
        return self.send(
            text=f"Metric `{metric_name}` exceeded threshold",
            level=level,
            title="Threshold Alert",
            fields={
                "Metric": metric_name,
                "Current": str(current_value),
                "Threshold": str(threshold),
            },
        )


# 싱글톤 인스턴스
_slack_notifier: Optional[SlackNotifier] = None


def get_slack_notifier(webhook_url: Optional[str] = None) -> SlackNotifier:
    """Slack 알림기 싱글톤 반환"""
    global _slack_notifier
    if _slack_notifier is None:
        _slack_notifier = SlackNotifier(webhook_url)
    return _slack_notifier


# 편의 함수
def notify_slack(
    text: str,
    level: AlertLevel = AlertLevel.INFO,
    **kwargs,
) -> bool:
    """Slack 알림 전송 편의 함수"""
    return get_slack_notifier().send(text, level, **kwargs)
