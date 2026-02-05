"""
Error Classification and Handling

오류 분류 및 처리 시스템
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type
import traceback
import hashlib

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class ErrorType(str, Enum):
    """오류 유형 분류"""
    # 네트워크 관련
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_CONNECTION = "network_connection"
    NETWORK_DNS = "network_dns"
    NETWORK_SSL = "network_ssl"
    
    # 인증/권한
    AUTH_UNAUTHORIZED = "auth_unauthorized"
    AUTH_FORBIDDEN = "auth_forbidden"
    AUTH_TOKEN_EXPIRED = "auth_token_expired"
    
    # API 관련
    API_RATE_LIMIT = "api_rate_limit"
    API_NOT_FOUND = "api_not_found"
    API_SERVER_ERROR = "api_server_error"
    API_BAD_REQUEST = "api_bad_request"
    
    # 데이터 관련
    DATA_VALIDATION = "data_validation"
    DATA_PARSING = "data_parsing"
    DATA_CORRUPTION = "data_corruption"
    DATA_MISSING = "data_missing"
    DATA_TYPE_ERROR = "data_type_error"
    
    # 스토리지 관련
    STORAGE_FULL = "storage_full"
    STORAGE_PERMISSION = "storage_permission"
    STORAGE_NOT_FOUND = "storage_not_found"
    STORAGE_IO = "storage_io"
    
    # 리소스 관련
    RESOURCE_MEMORY = "resource_memory"
    RESOURCE_CPU = "resource_cpu"
    RESOURCE_GPU = "resource_gpu"
    RESOURCE_TIMEOUT = "resource_timeout"
    
    # 시스템 관련
    SYSTEM_OS = "system_os"
    SYSTEM_DEPENDENCY = "system_dependency"
    SYSTEM_CONFIG = "system_config"
    
    # 비즈니스 로직
    BUSINESS_LOGIC = "business_logic"
    BUSINESS_CONSTRAINT = "business_constraint"
    
    # 알 수 없음
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """오류 심각도"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Retryable(str, Enum):
    """재시도 가능 여부"""
    YES = "yes"
    NO = "no"
    CONDITIONAL = "conditional"


@dataclass
class ErrorInfo:
    """오류 정보"""
    error_type: ErrorType
    severity: Severity
    retryable: Retryable
    message: str
    original_exception: Optional[Exception] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_id: Optional[str] = None
    
    def __post_init__(self):
        if self.error_id is None:
            self.error_id = self._generate_error_id()
    
    def _generate_error_id(self) -> str:
        """고유 에러 ID 생성"""
        content = f"{self.error_type.value}:{self.message}:{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "retryable": self.retryable.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "traceback": self.traceback,
        }


# 예외 유형별 분류 매핑
EXCEPTION_MAPPING: Dict[Type[Exception], Dict[str, Any]] = {
    # 네트워크 예외
    TimeoutError: {
        "error_type": ErrorType.NETWORK_TIMEOUT,
        "severity": Severity.WARNING,
        "retryable": Retryable.YES,
    },
    ConnectionError: {
        "error_type": ErrorType.NETWORK_CONNECTION,
        "severity": Severity.WARNING,
        "retryable": Retryable.YES,
    },
    ConnectionRefusedError: {
        "error_type": ErrorType.NETWORK_CONNECTION,
        "severity": Severity.ERROR,
        "retryable": Retryable.YES,
    },
    ConnectionResetError: {
        "error_type": ErrorType.NETWORK_CONNECTION,
        "severity": Severity.WARNING,
        "retryable": Retryable.YES,
    },
    
    # 데이터 예외
    ValueError: {
        "error_type": ErrorType.DATA_VALIDATION,
        "severity": Severity.WARNING,
        "retryable": Retryable.NO,
    },
    TypeError: {
        "error_type": ErrorType.DATA_TYPE_ERROR,
        "severity": Severity.ERROR,
        "retryable": Retryable.NO,
    },
    KeyError: {
        "error_type": ErrorType.DATA_MISSING,
        "severity": Severity.WARNING,
        "retryable": Retryable.NO,
    },
    
    # 파일/스토리지 예외
    FileNotFoundError: {
        "error_type": ErrorType.STORAGE_NOT_FOUND,
        "severity": Severity.WARNING,
        "retryable": Retryable.NO,
    },
    PermissionError: {
        "error_type": ErrorType.STORAGE_PERMISSION,
        "severity": Severity.ERROR,
        "retryable": Retryable.NO,
    },
    IOError: {
        "error_type": ErrorType.STORAGE_IO,
        "severity": Severity.ERROR,
        "retryable": Retryable.CONDITIONAL,
    },
    OSError: {
        "error_type": ErrorType.SYSTEM_OS,
        "severity": Severity.ERROR,
        "retryable": Retryable.CONDITIONAL,
    },
    
    # 리소스 예외
    MemoryError: {
        "error_type": ErrorType.RESOURCE_MEMORY,
        "severity": Severity.CRITICAL,
        "retryable": Retryable.CONDITIONAL,
    },
}


def classify_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    include_traceback: bool = True,
) -> ErrorInfo:
    """
    예외를 분류하여 ErrorInfo 반환
    
    Args:
        exception: 분류할 예외
        context: 추가 컨텍스트 정보
        include_traceback: 트레이스백 포함 여부
        
    Returns:
        ErrorInfo: 분류된 오류 정보
    """
    context = context or {}
    exc_type = type(exception)
    
    # 정확한 타입 매핑 확인
    if exc_type in EXCEPTION_MAPPING:
        mapping = EXCEPTION_MAPPING[exc_type]
    else:
        # 상위 클래스 확인
        mapping = None
        for base_type, base_mapping in EXCEPTION_MAPPING.items():
            if isinstance(exception, base_type):
                mapping = base_mapping
                break
        
        if mapping is None:
            # 기본값
            mapping = {
                "error_type": ErrorType.UNKNOWN,
                "severity": Severity.ERROR,
                "retryable": Retryable.NO,
            }
    
    # HTTP 예외 처리 (requests, httpx 등)
    error_type = mapping["error_type"]
    severity = mapping["severity"]
    retryable = mapping["retryable"]
    
    # HTTP 상태 코드 기반 분류
    status_code = getattr(exception, "status_code", None) or context.get("status_code")
    if status_code:
        error_type, severity, retryable = _classify_http_error(status_code)
    
    # 에러 메시지에서 추가 분류
    message = str(exception)
    if "rate limit" in message.lower():
        error_type = ErrorType.API_RATE_LIMIT
        retryable = Retryable.YES
    elif "timeout" in message.lower():
        error_type = ErrorType.NETWORK_TIMEOUT
        retryable = Retryable.YES
    elif "dns" in message.lower():
        error_type = ErrorType.NETWORK_DNS
        retryable = Retryable.YES
    elif "ssl" in message.lower() or "certificate" in message.lower():
        error_type = ErrorType.NETWORK_SSL
        retryable = Retryable.NO
    
    # 트레이스백 생성
    tb = None
    if include_traceback:
        tb = traceback.format_exc()
    
    return ErrorInfo(
        error_type=error_type,
        severity=severity,
        retryable=retryable,
        message=message,
        original_exception=exception,
        traceback=tb,
        context=context,
    )


def _classify_http_error(status_code: int) -> tuple:
    """HTTP 상태 코드 기반 분류"""
    if status_code == 401:
        return ErrorType.AUTH_UNAUTHORIZED, Severity.ERROR, Retryable.NO
    elif status_code == 403:
        return ErrorType.AUTH_FORBIDDEN, Severity.ERROR, Retryable.NO
    elif status_code == 404:
        return ErrorType.API_NOT_FOUND, Severity.WARNING, Retryable.NO
    elif status_code == 429:
        return ErrorType.API_RATE_LIMIT, Severity.WARNING, Retryable.YES
    elif status_code == 400:
        return ErrorType.API_BAD_REQUEST, Severity.WARNING, Retryable.NO
    elif 500 <= status_code < 600:
        return ErrorType.API_SERVER_ERROR, Severity.ERROR, Retryable.YES
    else:
        return ErrorType.UNKNOWN, Severity.ERROR, Retryable.NO


class ErrorRegistry:
    """오류 레지스트리 - 오류 집계 및 분석"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self._errors: List[ErrorInfo] = []
        self._error_counts: Dict[ErrorType, int] = {}
        
    def record(self, error: ErrorInfo):
        """오류 기록"""
        self._errors.append(error)
        
        # 카운트 증가
        if error.error_type not in self._error_counts:
            self._error_counts[error.error_type] = 0
        self._error_counts[error.error_type] += 1
        
        # 최대 개수 제한
        if len(self._errors) > self.max_errors:
            oldest = self._errors.pop(0)
            if self._error_counts.get(oldest.error_type, 0) > 0:
                self._error_counts[oldest.error_type] -= 1
    
    def get_counts(self) -> Dict[str, int]:
        """유형별 오류 수 반환"""
        return {k.value: v for k, v in self._error_counts.items()}
    
    def get_recent(self, limit: int = 10) -> List[ErrorInfo]:
        """최근 오류 반환"""
        return self._errors[-limit:]
    
    def get_by_type(self, error_type: ErrorType) -> List[ErrorInfo]:
        """유형별 오류 반환"""
        return [e for e in self._errors if e.error_type == error_type]
    
    def get_critical(self) -> List[ErrorInfo]:
        """크리티컬 오류 반환"""
        return [e for e in self._errors if e.severity == Severity.CRITICAL]
    
    def clear(self):
        """오류 기록 초기화"""
        self._errors.clear()
        self._error_counts.clear()
    
    def summary(self) -> Dict[str, Any]:
        """오류 요약"""
        return {
            "total_errors": len(self._errors),
            "by_type": self.get_counts(),
            "by_severity": self._count_by_severity(),
            "retryable_count": sum(1 for e in self._errors if e.retryable == Retryable.YES),
        }
    
    def _count_by_severity(self) -> Dict[str, int]:
        counts = {}
        for error in self._errors:
            sev = error.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts


# 전역 오류 레지스트리
_error_registry: Optional[ErrorRegistry] = None


def get_error_registry() -> ErrorRegistry:
    """오류 레지스트리 싱글톤 반환"""
    global _error_registry
    if _error_registry is None:
        _error_registry = ErrorRegistry()
    return _error_registry


def handle_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
) -> ErrorInfo:
    """
    오류 처리 헬퍼 함수
    
    Args:
        exception: 처리할 예외
        context: 추가 컨텍스트
        reraise: 예외 재발생 여부
        
    Returns:
        ErrorInfo: 분류된 오류 정보
    """
    error_info = classify_error(exception, context)
    
    # 레지스트리에 기록
    registry = get_error_registry()
    registry.record(error_info)
    
    # 로깅
    log_method = getattr(logger, error_info.severity.value, logger.error)
    log_method(
        f"[{error_info.error_id}] {error_info.error_type.value}: {error_info.message}",
        extra={"error_info": error_info.to_dict()},
    )
    
    if reraise:
        raise exception
    
    return error_info
