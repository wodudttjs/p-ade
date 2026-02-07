"""
Structured Logging Module

JSON 구조화 로깅 시스템
"""

from structured_logging.json_logger import (
    # 컨텍스트 함수
    set_request_id,
    get_request_id,
    set_job_id,
    get_job_id,
    set_user_id,
    get_user_id,
    set_stage,
    get_stage,
    set_extra_context,
    get_extra_context,
    generate_request_id,
    # 필터 및 포매터
    ContextFilter,
    JSONFormatter,
    PrettyJSONFormatter,
    # 설정 함수
    configure_json_logging,
    # 컨텍스트 관리자
    LogContext,
    # 로거
    StructuredLogger,
    get_structured_logger,
)

__all__ = [
    # 컨텍스트 함수
    "set_request_id",
    "get_request_id",
    "set_job_id",
    "get_job_id",
    "set_user_id",
    "get_user_id",
    "set_stage",
    "get_stage",
    "set_extra_context",
    "get_extra_context",
    "generate_request_id",
    # 필터 및 포매터
    "ContextFilter",
    "JSONFormatter",
    "PrettyJSONFormatter",
    # 설정 함수
    "configure_json_logging",
    # 컨텍스트 관리자
    "LogContext",
    # 로거
    "StructuredLogger",
    "get_structured_logger",
]
