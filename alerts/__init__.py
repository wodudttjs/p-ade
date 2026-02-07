"""
Alerts Module

알림 시스템 (Slack, Email, Manager)
"""

from alerts.slack import (
    AlertLevel,
    SlackAttachment,
    SlackMessage,
    SlackNotifier,
    get_slack_notifier,
    notify_slack,
)

from alerts.email import (
    EmailPriority,
    EmailConfig,
    EmailMessage,
    EmailNotifier,
    get_email_notifier,
    notify_email,
)

from alerts.manager import (
    AlertChannel,
    AlertState,
    AlertRule,
    Alert,
    AlertManager,
    get_alert_manager,
    register_default_rules,
)

__all__ = [
    # slack
    "AlertLevel",
    "SlackAttachment",
    "SlackMessage",
    "SlackNotifier",
    "get_slack_notifier",
    "notify_slack",
    # email
    "EmailPriority",
    "EmailConfig",
    "EmailMessage",
    "EmailNotifier",
    "get_email_notifier",
    "notify_email",
    # manager
    "AlertChannel",
    "AlertState",
    "AlertRule",
    "Alert",
    "AlertManager",
    "get_alert_manager",
    "register_default_rules",
]
