"""
JSON Structured Logging

구조화된 JSON 로깅 시스템
"""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from core.logging_config import setup_logger

# 컨텍스트 변수
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_job_id: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_stage: ContextVar[Optional[str]] = ContextVar("stage", default=None)
_extra_context: ContextVar[Dict[str, Any]] = ContextVar("extra_context", default={})


def set_request_id(request_id: str):
    """요청 ID 설정"""
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """요청 ID 조회"""
    return _request_id.get()


def set_job_id(job_id: str):
    """작업 ID 설정"""
    _job_id.set(job_id)


def get_job_id() -> Optional[str]:
    """작업 ID 조회"""
    return _job_id.get()


def set_user_id(user_id: str):
    """사용자 ID 설정"""
    _user_id.set(user_id)


def get_user_id() -> Optional[str]:
    """사용자 ID 조회"""
    return _user_id.get()


def set_stage(stage: str):
    """단계 설정"""
    _stage.set(stage)


def get_stage() -> Optional[str]:
    """단계 조회"""
    return _stage.get()


def set_extra_context(context: Dict[str, Any]):
    """추가 컨텍스트 설정"""
    _extra_context.set(context)


def get_extra_context() -> Dict[str, Any]:
    """추가 컨텍스트 조회"""
    return _extra_context.get()


def generate_request_id() -> str:
    """새 요청 ID 생성"""
    return str(uuid.uuid4())


class ContextFilter(logging.Filter):
    """로그에 컨텍스트 정보를 추가하는 필터"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        record.job_id = get_job_id() or "-"
        record.user_id = get_user_id() or "-"
        record.stage = get_stage() or "-"
        record.extra_context = get_extra_context()
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON 형식 로그 포매터
    
    Task 6.2.1: 구조화된 JSON 로깅
    """
    
    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    }
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_path: bool = False,
        include_function: bool = True,
        include_process: bool = False,
        include_thread: bool = False,
        extra_fields: Optional[List[str]] = None,
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_path = include_path
        self.include_function = include_function
        self.include_process = include_process
        self.include_thread = include_thread
        self.extra_fields = extra_fields or []
        self.timestamp_format = timestamp_format
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {}
        
        # 타임스탬프
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().strftime(self.timestamp_format)
        
        # 레벨
        if self.include_level:
            log_data["level"] = record.levelname
        
        # 로거 이름
        if self.include_logger:
            log_data["logger"] = record.name
        
        # 메시지
        log_data["message"] = record.getMessage()
        
        # 위치 정보
        if self.include_path:
            log_data["path"] = record.pathname
            log_data["line"] = record.lineno
        
        if self.include_function:
            log_data["function"] = record.funcName
        
        # 프로세스/스레드 정보
        if self.include_process:
            log_data["process"] = {
                "id": record.process,
                "name": record.processName,
            }
        
        if self.include_thread:
            log_data["thread"] = {
                "id": record.thread,
                "name": record.threadName,
            }
        
        # 컨텍스트 정보 (ContextFilter에서 추가)
        if hasattr(record, "request_id") and record.request_id != "-":
            log_data["request_id"] = record.request_id
        if hasattr(record, "job_id") and record.job_id != "-":
            log_data["job_id"] = record.job_id
        if hasattr(record, "user_id") and record.user_id != "-":
            log_data["user_id"] = record.user_id
        if hasattr(record, "stage") and record.stage != "-":
            log_data["stage"] = record.stage
        if hasattr(record, "extra_context") and record.extra_context:
            log_data.update(record.extra_context)
        
        # 추가 필드 (record에서)
        for field in self.extra_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        # extra 딕셔너리에서 추가 필드 추출
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                if key not in log_data:
                    log_data[key] = value
        
        # 예외 정보
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class PrettyJSONFormatter(JSONFormatter):
    """들여쓰기된 JSON 포매터 (개발용)"""
    
    def format(self, record: logging.LogRecord) -> str:
        json_str = super().format(record)
        log_data = json.loads(json_str)
        return json.dumps(log_data, indent=2, ensure_ascii=False, default=str)


def configure_json_logging(
    level: int = logging.INFO,
    output: str = "stdout",
    log_file: Optional[str] = None,
    pretty: bool = False,
) -> logging.Logger:
    """
    JSON 로깅 설정
    
    Args:
        level: 로그 레벨
        output: 출력 대상 ("stdout", "stderr", "file")
        log_file: 로그 파일 경로 (output="file"일 때)
        pretty: 들여쓰기 포맷 사용 여부
    
    Returns:
        설정된 루트 로거
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 포매터 선택
    formatter = PrettyJSONFormatter() if pretty else JSONFormatter()
    
    # 핸들러 설정
    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif output == "file" and log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    
    root_logger.addHandler(handler)
    
    return root_logger


class LogContext:
    """로그 컨텍스트 관리자"""
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stage: Optional[str] = None,
        **extra,
    ):
        self.request_id = request_id
        self.job_id = job_id
        self.user_id = user_id
        self.stage = stage
        self.extra = extra
        self._prev_request_id = None
        self._prev_job_id = None
        self._prev_user_id = None
        self._prev_stage = None
        self._prev_extra = None
    
    def __enter__(self):
        # 이전 값 저장
        self._prev_request_id = get_request_id()
        self._prev_job_id = get_job_id()
        self._prev_user_id = get_user_id()
        self._prev_stage = get_stage()
        self._prev_extra = get_extra_context()
        
        # 새 값 설정
        if self.request_id:
            set_request_id(self.request_id)
        if self.job_id:
            set_job_id(self.job_id)
        if self.user_id:
            set_user_id(self.user_id)
        if self.stage:
            set_stage(self.stage)
        if self.extra:
            set_extra_context({**self._prev_extra, **self.extra})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 이전 값 복원
        if self._prev_request_id is not None:
            set_request_id(self._prev_request_id)
        if self._prev_job_id is not None:
            set_job_id(self._prev_job_id)
        if self._prev_user_id is not None:
            set_user_id(self._prev_user_id)
        if self._prev_stage is not None:
            set_stage(self._prev_stage)
        if self._prev_extra is not None:
            set_extra_context(self._prev_extra)
        
        return False


class StructuredLogger:
    """구조화된 로거 래퍼"""
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
    
    def _log(self, level: int, message: str, **kwargs):
        extra = kwargs.pop("extra", {})
        extra.update(kwargs)
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)
    
    def event(self, event_name: str, **kwargs):
        """이벤트 로깅"""
        kwargs["event"] = event_name
        self._log(logging.INFO, f"Event: {event_name}", **kwargs)
    
    def metric(self, metric_name: str, value: float, **kwargs):
        """메트릭 로깅"""
        kwargs["metric_name"] = metric_name
        kwargs["metric_value"] = value
        self._log(logging.INFO, f"Metric: {metric_name}={value}", **kwargs)


def get_structured_logger(name: str) -> StructuredLogger:
    """구조화된 로거 반환"""
    return StructuredLogger(name)
