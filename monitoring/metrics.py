"""
Prometheus Metrics Collection

파이프라인 메트릭 수집
- 처리량, 오류율, 지연 시간
- 리소스 사용량 (CPU, GPU, 메모리)
- 큐 상태
"""

import time
import functools
from typing import Optional, Dict, Any, Callable
from enum import Enum
from contextlib import contextmanager

from core.logging_config import setup_logger

logger = setup_logger(__name__)

# Prometheus 클라이언트 lazy import
_prometheus_available = None
_metrics_registry = {}


def _check_prometheus():
    """Prometheus 클라이언트 가용성 확인"""
    global _prometheus_available
    if _prometheus_available is None:
        try:
            import prometheus_client
            _prometheus_available = True
        except ImportError:
            _prometheus_available = False
            logger.warning("prometheus_client not installed. Metrics will be no-op.")
    return _prometheus_available


class MetricType(Enum):
    """메트릭 유형"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Stage(Enum):
    """파이프라인 스테이지"""
    DISCOVER = "discover"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    UPLOAD = "upload"
    VERSION_FINALIZE = "version_finalize"


class Status(Enum):
    """작업 상태"""
    SUCCESS = "success"
    FAIL = "fail"
    SKIP = "skip"


class ItemType(Enum):
    """아이템 유형"""
    VIDEO = "video"
    EPISODE = "episode"
    FILE = "file"


# 메트릭 버킷 정의
DURATION_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class MetricsRegistry:
    """
    Prometheus 메트릭 레지스트리
    
    Task 6.1.1: Metrics Collection
    """
    
    def __init__(self, prefix: str = "pipeline_"):
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize(self):
        """메트릭 초기화"""
        if self._initialized or not _check_prometheus():
            return
            
        from prometheus_client import Counter, Gauge, Histogram
        
        # ===== 처리량/성공률 =====
        self._metrics["jobs_total"] = Counter(
            f"{self.prefix}jobs_total",
            "Total number of jobs processed",
            ["stage", "status"],
        )
        
        self._metrics["items_processed_total"] = Counter(
            f"{self.prefix}items_processed_total",
            "Total items processed",
            ["stage", "item_type"],
        )
        
        # ===== 지연/성능 =====
        self._metrics["job_duration_seconds"] = Histogram(
            f"{self.prefix}job_duration_seconds",
            "Job duration in seconds",
            ["stage"],
            buckets=DURATION_BUCKETS,
        )
        
        self._metrics["queue_latency_seconds"] = Histogram(
            f"{self.prefix}queue_latency_seconds",
            "Queue latency (enqueue to start)",
            ["stage"],
            buckets=LATENCY_BUCKETS,
        )
        
        # ===== 오류율 =====
        self._metrics["errors_total"] = Counter(
            f"{self.prefix}errors_total",
            "Total number of errors",
            ["stage", "error_type"],
        )
        
        # ===== 큐/백로그 =====
        self._metrics["queue_depth"] = Gauge(
            f"{self.prefix}queue_depth",
            "Current queue depth",
            ["queue_name"],
        )
        
        self._metrics["inflight_jobs"] = Gauge(
            f"{self.prefix}inflight_jobs",
            "Currently in-flight jobs",
            ["stage"],
        )
        
        # ===== 리소스 사용량 =====
        self._metrics["cpu_percent"] = Gauge(
            f"{self.prefix}cpu_percent",
            "CPU usage percentage",
        )
        
        self._metrics["memory_bytes"] = Gauge(
            f"{self.prefix}memory_bytes",
            "Memory usage in bytes",
        )
        
        self._metrics["memory_percent"] = Gauge(
            f"{self.prefix}memory_percent",
            "Memory usage percentage",
        )
        
        self._metrics["gpu_util_percent"] = Gauge(
            f"{self.prefix}gpu_util_percent",
            "GPU utilization percentage",
            ["gpu_id"],
        )
        
        self._metrics["gpu_memory_bytes"] = Gauge(
            f"{self.prefix}gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["gpu_id"],
        )
        
        # ===== 스토리지/비용 =====
        self._metrics["storage_bytes"] = Gauge(
            f"{self.prefix}storage_bytes",
            "Total storage bytes",
            ["provider", "storage_class"],
        )
        
        self._metrics["storage_cost_usd_monthly"] = Gauge(
            f"{self.prefix}storage_cost_usd_monthly_estimate",
            "Estimated monthly storage cost in USD",
            ["provider"],
        )
        
        self._initialized = True
        logger.info("Prometheus metrics initialized")
        
    def get_metric(self, name: str) -> Optional[Any]:
        """메트릭 가져오기"""
        if not self._initialized:
            self.initialize()
        return self._metrics.get(name)
    
    # ===== 처리량 메트릭 =====
    
    def inc_jobs(self, stage: str, status: str, value: int = 1):
        """작업 카운터 증가"""
        metric = self.get_metric("jobs_total")
        if metric:
            metric.labels(stage=stage, status=status).inc(value)
            
    def inc_items(self, stage: str, item_type: str, value: int = 1):
        """아이템 카운터 증가"""
        metric = self.get_metric("items_processed_total")
        if metric:
            metric.labels(stage=stage, item_type=item_type).inc(value)
            
    def inc_errors(self, stage: str, error_type: str, value: int = 1):
        """오류 카운터 증가"""
        metric = self.get_metric("errors_total")
        if metric:
            metric.labels(stage=stage, error_type=error_type).inc(value)
    
    # ===== 지연 메트릭 =====
    
    def observe_duration(self, stage: str, duration: float):
        """작업 시간 기록"""
        metric = self.get_metric("job_duration_seconds")
        if metric:
            metric.labels(stage=stage).observe(duration)
            
    def observe_queue_latency(self, stage: str, latency: float):
        """큐 대기 시간 기록"""
        metric = self.get_metric("queue_latency_seconds")
        if metric:
            metric.labels(stage=stage).observe(latency)
    
    # ===== 큐 메트릭 =====
    
    def set_queue_depth(self, queue_name: str, depth: int):
        """큐 깊이 설정"""
        metric = self.get_metric("queue_depth")
        if metric:
            metric.labels(queue_name=queue_name).set(depth)
            
    def set_inflight(self, stage: str, count: int):
        """진행 중 작업 수 설정"""
        metric = self.get_metric("inflight_jobs")
        if metric:
            metric.labels(stage=stage).set(count)
            
    def inc_inflight(self, stage: str):
        """진행 중 작업 수 증가"""
        metric = self.get_metric("inflight_jobs")
        if metric:
            metric.labels(stage=stage).inc()
            
    def dec_inflight(self, stage: str):
        """진행 중 작업 수 감소"""
        metric = self.get_metric("inflight_jobs")
        if metric:
            metric.labels(stage=stage).dec()
    
    # ===== 리소스 메트릭 =====
    
    def update_cpu(self, percent: float):
        """CPU 사용량 업데이트"""
        metric = self.get_metric("cpu_percent")
        if metric:
            metric.set(percent)
            
    def update_memory(self, bytes_used: int, percent: float):
        """메모리 사용량 업데이트"""
        mem_bytes = self.get_metric("memory_bytes")
        mem_pct = self.get_metric("memory_percent")
        if mem_bytes:
            mem_bytes.set(bytes_used)
        if mem_pct:
            mem_pct.set(percent)
            
    def update_gpu(self, gpu_id: str, util_percent: float, memory_bytes: int):
        """GPU 사용량 업데이트"""
        gpu_util = self.get_metric("gpu_util_percent")
        gpu_mem = self.get_metric("gpu_memory_bytes")
        if gpu_util:
            gpu_util.labels(gpu_id=gpu_id).set(util_percent)
        if gpu_mem:
            gpu_mem.labels(gpu_id=gpu_id).set(memory_bytes)
    
    # ===== 스토리지 메트릭 =====
    
    def update_storage(self, provider: str, storage_class: str, bytes_total: int):
        """스토리지 용량 업데이트"""
        metric = self.get_metric("storage_bytes")
        if metric:
            metric.labels(provider=provider, storage_class=storage_class).set(bytes_total)
            
    def update_storage_cost(self, provider: str, cost_usd: float):
        """스토리지 비용 업데이트"""
        metric = self.get_metric("storage_cost_usd_monthly")
        if metric:
            metric.labels(provider=provider).set(cost_usd)


# 전역 레지스트리 인스턴스
metrics = MetricsRegistry()


def observe_job(stage: str):
    """
    작업 관측 데코레이터
    
    사용법:
        @observe_job(stage="transform")
        def process_video(video_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics.inc_inflight(stage)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                metrics.inc_jobs(stage, Status.SUCCESS.value)
                return result
                
            except Exception as e:
                from errors.errors import classify_error
                error_info = classify_error(e)
                metrics.inc_errors(stage, error_info.error_type.value)
                metrics.inc_jobs(stage, Status.FAIL.value)
                raise
                
            finally:
                duration = time.time() - start_time
                metrics.observe_duration(stage, duration)
                metrics.dec_inflight(stage)
                
        return wrapper
    return decorator


@contextmanager
def observe_job_context(stage: str):
    """
    작업 관측 컨텍스트 매니저
    
    사용법:
        with observe_job_context("download"):
            download_video(url)
    """
    metrics.inc_inflight(stage)
    start_time = time.time()
    
    try:
        yield
        metrics.inc_jobs(stage, Status.SUCCESS.value)
        
    except Exception as e:
        from errors.errors import classify_error
        error_info = classify_error(e)
        metrics.inc_errors(stage, error_info.error_type.value)
        metrics.inc_jobs(stage, Status.FAIL.value)
        raise
        
    finally:
        duration = time.time() - start_time
        metrics.observe_duration(stage, duration)
        metrics.dec_inflight(stage)


class ResourceCollector:
    """
    리소스 사용량 수집기
    
    CPU, 메모리, GPU 사용량을 주기적으로 수집
    """
    
    def __init__(self, metrics_registry: MetricsRegistry, interval_seconds: float = 15.0):
        self.metrics = metrics_registry
        self.interval = interval_seconds
        self._running = False
        self._thread = None
    
    def _collect_cpu(self) -> float:
        """CPU 사용률 수집"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _collect_memory(self) -> int:
        """메모리 사용량 수집 (bytes)"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.used
        except ImportError:
            return 0
        
    def collect_once(self):
        """리소스 사용량 한 번 수집"""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent()
            self.metrics.update_cpu(cpu_percent)
            
            # 메모리
            mem = psutil.virtual_memory()
            self.metrics.update_memory(mem.used, mem.percent)
            
        except ImportError:
            pass
            
        # GPU (NVML)
        self._collect_gpu()
        
    def _collect_gpu(self):
        """GPU 사용량 수집 (NVML)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 사용률
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 메모리
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                self.metrics.update_gpu(
                    gpu_id=str(i),
                    util_percent=util.gpu,
                    memory_bytes=mem.used,
                )
                
            pynvml.nvmlShutdown()
            
        except (ImportError, Exception):
            # NVML 사용 불가
            pass
    
    def start(self):
        """백그라운드 수집 시작"""
        self.start_background(interval_sec=self.interval)
    
    def start_background(self, interval_sec: float = 15.0):
        """백그라운드 수집 시작"""
        import threading
        
        def collector_loop():
            while self._running:
                self.collect_once()
                time.sleep(interval_sec)
                
        self._running = True
        self._thread = threading.Thread(target=collector_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """백그라운드 수집 중지"""
        self._running = False
