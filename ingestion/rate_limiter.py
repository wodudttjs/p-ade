"""
레이트 리미터 & 재시도 매니저

대량 수집 시 API/사이트 차단을 방지하기 위한 레이트 리미팅,
지수 백오프 재시도, 프록시 로테이션 기능을 제공합니다.
"""

import time
import random
import threading
from collections import defaultdict
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class RateLimitConfig:
    """레이트 리밋 설정"""
    requests_per_minute: int = 30       # 분당 최대 요청
    requests_per_hour: int = 500        # 시간당 최대 요청
    min_delay_sec: float = 1.0          # 최소 요청 간격 (초)
    max_delay_sec: float = 5.0          # 최대 요청 간격 (초)
    jitter_range: float = 0.5           # 지터 범위 (초)
    burst_limit: int = 10               # 버스트 제한
    burst_window_sec: float = 10.0      # 버스트 윈도우 (초)


@dataclass
class RetryConfig:
    """재시도 설정"""
    max_retries: int = 5
    base_delay_sec: float = 2.0          # 기본 대기 시간
    max_delay_sec: float = 300.0         # 최대 대기 시간 (5분)
    exponential_base: float = 2.0        # 지수 백오프 기수
    jitter: bool = True                  # 랜덤 지터 사용
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class TokenBucketLimiter:
    """토큰 버킷 알고리즘 기반 레이트 리미터"""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()
        self._tokens = self.config.burst_limit
        self._last_refill = time.monotonic()
        self._request_timestamps: List[float] = []

        # 분당 요청 → 초당 리필율
        self._refill_rate = self.config.requests_per_minute / 60.0

    def acquire(self, timeout: float = 60.0) -> bool:
        """토큰 획득 (True: 진행 가능, False: 타임아웃)"""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    self._record_request()
                    return True

            # 대기
            wait = 1.0 / max(self._refill_rate, 0.1)
            wait += random.uniform(0, self.config.jitter_range)
            time.sleep(min(wait, deadline - time.monotonic()))

        logger.warning("레이트 리미터: 타임아웃으로 토큰 획득 실패")
        return False

    def wait(self):
        """다음 요청까지 대기 (블로킹)"""
        delay = random.uniform(
            self.config.min_delay_sec,
            self.config.max_delay_sec,
        )
        delay += random.uniform(0, self.config.jitter_range)
        time.sleep(delay)

    def _refill(self):
        """토큰 리필"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(
            self._tokens + new_tokens,
            self.config.burst_limit,
        )
        self._last_refill = now

    def _record_request(self):
        """요청 기록"""
        now = time.monotonic()
        self._request_timestamps.append(now)
        # 1시간 이전 기록 제거
        cutoff = now - 3600
        self._request_timestamps = [
            t for t in self._request_timestamps if t > cutoff
        ]

    @property
    def requests_last_minute(self) -> int:
        """최근 1분간 요청 수"""
        cutoff = time.monotonic() - 60
        return sum(1 for t in self._request_timestamps if t > cutoff)

    @property
    def requests_last_hour(self) -> int:
        """최근 1시간 요청 수"""
        cutoff = time.monotonic() - 3600
        return sum(1 for t in self._request_timestamps if t > cutoff)

    def is_rate_limited(self) -> bool:
        """현재 레이트 리밋 상태인지 확인"""
        if self.requests_last_minute >= self.config.requests_per_minute:
            return True
        if self.requests_last_hour >= self.config.requests_per_hour:
            return True
        return False


class RetryManager:
    """지수 백오프 재시도 매니저"""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._retry_counts: Dict[str, int] = defaultdict(int)
        self._total_retries = 0
        self._total_failures = 0

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        task_id: str = "",
        **kwargs,
    ) -> Any:
        """함수를 재시도 로직과 함께 실행"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"재시도 성공: {task_id} (시도 {attempt + 1}/{self.config.max_retries + 1})"
                    )
                return result

            except self.config.retryable_exceptions as e:
                last_exception = e
                self._total_retries += 1

                if attempt >= self.config.max_retries:
                    self._total_failures += 1
                    logger.error(
                        f"재시도 한도 초과: {task_id} ({self.config.max_retries + 1}회 시도) - {e}"
                    )
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"재시도 대기: {task_id} (시도 {attempt + 1}, "
                    f"{delay:.1f}초 후 재시도) - {e}"
                )
                time.sleep(delay)

            except Exception as e:
                # 재시도 불가능한 예외
                self._total_failures += 1
                logger.error(f"재시도 불가: {task_id} - {type(e).__name__}: {e}")
                raise

        raise last_exception  # type: ignore

    def _calculate_delay(self, attempt: int) -> float:
        """지수 백오프 대기 시간 계산"""
        delay = self.config.base_delay_sec * (
            self.config.exponential_base ** attempt
        )
        delay = min(delay, self.config.max_delay_sec)

        if self.config.jitter:
            delay *= random.uniform(0.5, 1.5)

        return delay

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
        }


class SourceRateLimiter:
    """소스별 독립 레이트 리미터"""

    # 소스별 기본 설정
    SOURCE_CONFIGS = {
        "youtube": RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=300,
            min_delay_sec=2.0,
            max_delay_sec=5.0,
        ),
        "google_videos": RateLimitConfig(
            requests_per_minute=15,
            requests_per_hour=200,
            min_delay_sec=3.0,
            max_delay_sec=8.0,
        ),
        "vimeo": RateLimitConfig(
            requests_per_minute=25,
            requests_per_hour=400,
            min_delay_sec=1.5,
            max_delay_sec=4.0,
        ),
        "dailymotion": RateLimitConfig(
            requests_per_minute=25,
            requests_per_hour=400,
            min_delay_sec=1.5,
            max_delay_sec=4.0,
        ),
        "bilibili": RateLimitConfig(
            requests_per_minute=15,
            requests_per_hour=200,
            min_delay_sec=3.0,
            max_delay_sec=8.0,
        ),
        "rutube": RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=300,
            min_delay_sec=2.0,
            max_delay_sec=5.0,
        ),
    }

    def __init__(self):
        self._limiters: Dict[str, TokenBucketLimiter] = {}
        self._retry_manager = RetryManager()

    def get_limiter(self, source: str) -> TokenBucketLimiter:
        """소스별 리미터 반환 (없으면 생성)"""
        if source not in self._limiters:
            config = self.SOURCE_CONFIGS.get(source, RateLimitConfig())
            self._limiters[source] = TokenBucketLimiter(config)
        return self._limiters[source]

    def wait_for(self, source: str):
        """소스에 맞는 대기"""
        limiter = self.get_limiter(source)
        limiter.acquire()
        limiter.wait()

    def execute(
        self,
        source: str,
        func: Callable,
        *args,
        task_id: str = "",
        **kwargs,
    ) -> Any:
        """레이트 리밋 + 재시도와 함께 실행"""
        self.wait_for(source)
        return self._retry_manager.execute_with_retry(
            func, *args, task_id=task_id, **kwargs
        )

    def get_stats(self) -> Dict[str, Dict]:
        """전체 통계"""
        stats = {}
        for source, limiter in self._limiters.items():
            stats[source] = {
                "requests_last_minute": limiter.requests_last_minute,
                "requests_last_hour": limiter.requests_last_hour,
                "is_rate_limited": limiter.is_rate_limited(),
            }
        stats["retry"] = self._retry_manager.stats
        return stats


def rate_limited(source: str = "default"):
    """레이트 리밋 데코레이터"""
    _source_limiter = SourceRateLimiter()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _source_limiter.wait_for(source)
            return func(*args, **kwargs)
        return wrapper
    return decorator
