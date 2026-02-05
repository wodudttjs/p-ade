"""
Email Alert Notifier

이메일을 통한 알림 전송
"""

import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class EmailPriority(str, Enum):
    """이메일 우선순위"""
    LOW = "5"
    NORMAL = "3"
    HIGH = "1"


@dataclass
class EmailConfig:
    """이메일 설정"""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    from_address: Optional[str] = None
    from_name: str = "P-ADE Alert System"
    
    @classmethod
    def from_env(cls) -> "EmailConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            use_ssl=os.getenv("SMTP_USE_SSL", "false").lower() == "true",
            from_address=os.getenv("SMTP_FROM_ADDRESS"),
            from_name=os.getenv("SMTP_FROM_NAME", "P-ADE Alert System"),
        )


@dataclass
class EmailMessage:
    """이메일 메시지"""
    to: List[str]
    subject: str
    body_text: str
    body_html: Optional[str] = None
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    priority: EmailPriority = EmailPriority.NORMAL
    headers: Dict[str, str] = field(default_factory=dict)


class EmailNotifier:
    """
    이메일 알림 전송기
    
    Task 6.2.4: Email 알림 시스템
    """
    
    def __init__(
        self,
        config: Optional[EmailConfig] = None,
        enabled: bool = True,
    ):
        self.config = config or EmailConfig.from_env()
        self.enabled = enabled and self.config.smtp_user is not None
        
        if not self.enabled:
            logger.warning("Email notifications disabled (no SMTP credentials)")
    
    def _create_mime_message(
        self,
        message: EmailMessage,
    ) -> MIMEMultipart:
        """MIME 메시지 생성"""
        msg = MIMEMultipart("alternative")
        
        # 헤더 설정
        from_addr = self.config.from_address or self.config.smtp_user
        msg["From"] = f"{self.config.from_name} <{from_addr}>"
        msg["To"] = ", ".join(message.to)
        msg["Subject"] = message.subject
        msg["X-Priority"] = message.priority.value
        
        if message.cc:
            msg["Cc"] = ", ".join(message.cc)
        
        # 커스텀 헤더
        for key, value in message.headers.items():
            msg[key] = value
        
        # 본문 추가
        msg.attach(MIMEText(message.body_text, "plain", "utf-8"))
        
        if message.body_html:
            msg.attach(MIMEText(message.body_html, "html", "utf-8"))
        
        return msg
    
    def _send_email(self, message: EmailMessage) -> bool:
        """이메일 전송"""
        if not self.enabled:
            logger.debug(f"Email disabled, skipping: {message.subject}")
            return False
        
        try:
            # SMTP 연결
            if self.config.use_ssl:
                server = smtplib.SMTP_SSL(
                    self.config.smtp_host,
                    self.config.smtp_port,
                )
            else:
                server = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                )
                if self.config.use_tls:
                    server.starttls()
            
            # 인증
            if self.config.smtp_user and self.config.smtp_password:
                server.login(self.config.smtp_user, self.config.smtp_password)
            
            # 메시지 생성 및 전송
            mime_msg = self._create_mime_message(message)
            all_recipients = message.to + message.cc + message.bcc
            
            server.sendmail(
                self.config.from_address or self.config.smtp_user,
                all_recipients,
                mime_msg.as_string(),
            )
            
            server.quit()
            
            logger.info(f"Email sent: {message.subject} to {message.to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send(
        self,
        to: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        priority: EmailPriority = EmailPriority.NORMAL,
    ) -> bool:
        """
        이메일 전송
        
        Args:
            to: 수신자 목록
            subject: 제목
            body: 텍스트 본문
            html_body: HTML 본문 (선택)
            priority: 우선순위
        
        Returns:
            성공 여부
        """
        message = EmailMessage(
            to=to,
            subject=subject,
            body_text=body,
            body_html=html_body,
            priority=priority,
        )
        return self._send_email(message)
    
    def send_alert(
        self,
        to: List[str],
        alert_type: str,
        alert_message: str,
        details: Optional[Dict[str, Any]] = None,
        priority: EmailPriority = EmailPriority.HIGH,
    ) -> bool:
        """알림 이메일 전송"""
        subject = f"[P-ADE Alert] {alert_type}"
        
        # 텍스트 본문
        text_lines = [
            f"Alert Type: {alert_type}",
            f"Time: {datetime.utcnow().isoformat()}Z",
            "",
            "Message:",
            alert_message,
        ]
        
        if details:
            text_lines.append("")
            text_lines.append("Details:")
            for key, value in details.items():
                text_lines.append(f"  - {key}: {value}")
        
        body = "\n".join(text_lines)
        
        # HTML 본문
        html_body = self._generate_alert_html(
            alert_type, alert_message, details
        )
        
        return self.send(
            to=to,
            subject=subject,
            body=body,
            html_body=html_body,
            priority=priority,
        )
    
    def _generate_alert_html(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """알림 HTML 생성"""
        # 우선순위에 따른 색상
        color = "#dc3545" if "CRITICAL" in alert_type else "#ffc107"
        
        details_html = ""
        if details:
            rows = "".join(
                f"<tr><td style='padding:5px;font-weight:bold;'>{k}</td>"
                f"<td style='padding:5px;'>{v}</td></tr>"
                for k, v in details.items()
            )
            details_html = f"""
            <table style="border-collapse:collapse;margin-top:15px;">
                {rows}
            </table>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family:Arial,sans-serif;line-height:1.6;color:#333;">
            <div style="max-width:600px;margin:0 auto;padding:20px;">
                <div style="background-color:{color};color:white;padding:15px;border-radius:5px 5px 0 0;">
                    <h2 style="margin:0;">🚨 {alert_type}</h2>
                </div>
                <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 5px 5px;">
                    <p style="margin:0 0 15px 0;">
                        <strong>Time:</strong> {datetime.utcnow().isoformat()}Z
                    </p>
                    <p style="margin:0 0 15px 0;">
                        <strong>Message:</strong><br>
                        {message}
                    </p>
                    {details_html}
                </div>
                <div style="margin-top:20px;font-size:12px;color:#666;">
                    <p>This is an automated message from P-ADE Pipeline Alert System.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def send_error_report(
        self,
        to: List[str],
        error_type: str,
        error_message: str,
        error_id: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """오류 보고서 이메일 전송"""
        details = {
            "Error ID": error_id,
            "Error Type": error_type,
        }
        if context:
            details.update(context)
        
        message = error_message
        if traceback:
            message += f"\n\nTraceback:\n{traceback}"
        
        return self.send_alert(
            to=to,
            alert_type=f"ERROR - {error_type}",
            alert_message=message,
            details=details,
            priority=EmailPriority.HIGH,
        )
    
    def send_daily_summary(
        self,
        to: List[str],
        summary: Dict[str, Any],
    ) -> bool:
        """일일 요약 이메일 전송"""
        subject = f"[P-ADE] Daily Summary - {datetime.utcnow().date()}"
        
        # 텍스트 본문
        text_lines = [
            "P-ADE Pipeline Daily Summary",
            f"Date: {datetime.utcnow().date()}",
            "",
        ]
        
        for section, data in summary.items():
            text_lines.append(f"== {section} ==")
            if isinstance(data, dict):
                for key, value in data.items():
                    text_lines.append(f"  {key}: {value}")
            else:
                text_lines.append(f"  {data}")
            text_lines.append("")
        
        body = "\n".join(text_lines)
        
        return self.send(
            to=to,
            subject=subject,
            body=body,
            priority=EmailPriority.NORMAL,
        )


# 싱글톤 인스턴스
_email_notifier: Optional[EmailNotifier] = None


def get_email_notifier(config: Optional[EmailConfig] = None) -> EmailNotifier:
    """이메일 알림기 싱글톤 반환"""
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier(config)
    return _email_notifier


# 편의 함수
def notify_email(
    to: List[str],
    subject: str,
    body: str,
    **kwargs,
) -> bool:
    """이메일 알림 전송 편의 함수"""
    return get_email_notifier().send(to, subject, body, **kwargs)
