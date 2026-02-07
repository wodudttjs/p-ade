"""
KPI Dashboard Metrics

주요 성과 지표 계산 및 집계
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import statistics

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class KPIType(str, Enum):
    """KPI 유형"""
    THROUGHPUT = "throughput"  # 처리량
    LATENCY = "latency"  # 지연 시간
    ERROR_RATE = "error_rate"  # 에러율
    SUCCESS_RATE = "success_rate"  # 성공률
    AVAILABILITY = "availability"  # 가용성
    COST = "cost"  # 비용
    EFFICIENCY = "efficiency"  # 효율성


class Aggregation(str, Enum):
    """집계 방법"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"


@dataclass
class KPIMetric:
    """KPI 메트릭"""
    name: str
    kpi_type: KPIType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def status(self) -> str:
        """상태 판정"""
        if self.threshold_critical is not None:
            # error_rate, latency: 높을수록 나쁨
            if self.kpi_type in (KPIType.ERROR_RATE, KPIType.LATENCY, KPIType.COST):
                if self.value >= self.threshold_critical:
                    return "critical"
                if self.threshold_warning and self.value >= self.threshold_warning:
                    return "warning"
            # success_rate, throughput: 낮을수록 나쁨
            else:
                if self.value <= self.threshold_critical:
                    return "critical"
                if self.threshold_warning and self.value <= self.threshold_warning:
                    return "warning"
        return "ok"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kpi_type": self.kpi_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "status": self.status,
        }


@dataclass
class KPIDashboard:
    """KPI 대시보드 상태"""
    metrics: List[KPIMetric] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def add_metric(self, metric: KPIMetric):
        self.metrics.append(metric)
        
    def get_by_type(self, kpi_type: KPIType) -> List[KPIMetric]:
        return [m for m in self.metrics if m.kpi_type == kpi_type]
    
    def get_critical(self) -> List[KPIMetric]:
        return [m for m in self.metrics if m.status == "critical"]
    
    def get_warnings(self) -> List[KPIMetric]:
        return [m for m in self.metrics if m.status == "warning"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat() if self.period_start else None,
                "end": self.period_end.isoformat() if self.period_end else None,
            },
            "summary": {
                "total": len(self.metrics),
                "critical": len(self.get_critical()),
                "warning": len(self.get_warnings()),
            },
            "metrics": [m.to_dict() for m in self.metrics],
        }


class KPICalculator:
    """
    KPI 계산기
    
    Task 6.1.3: KPI 대시보드
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_time: Optional[datetime] = None
        
    def calculate_throughput(
        self,
        counts: List[int],
        duration_seconds: float,
        unit: str = "items/sec",
    ) -> float:
        """처리량 계산"""
        if duration_seconds <= 0:
            return 0.0
        total = sum(counts)
        return total / duration_seconds
    
    def calculate_latency_percentile(
        self,
        latencies: List[float],
        percentile: float = 0.95,
    ) -> float:
        """지연시간 백분위수 계산"""
        if not latencies:
            return 0.0
        sorted_lat = sorted(latencies)
        index = int(len(sorted_lat) * percentile)
        return sorted_lat[min(index, len(sorted_lat) - 1)]
    
    def calculate_error_rate(
        self,
        error_count: int,
        total_count: int,
    ) -> float:
        """에러율 계산"""
        if total_count <= 0:
            return 0.0
        return (error_count / total_count) * 100
    
    def calculate_success_rate(
        self,
        success_count: int,
        total_count: int,
    ) -> float:
        """성공률 계산"""
        if total_count <= 0:
            return 100.0
        return (success_count / total_count) * 100
    
    def calculate_availability(
        self,
        uptime_seconds: float,
        total_seconds: float,
    ) -> float:
        """가용성 계산"""
        if total_seconds <= 0:
            return 100.0
        return (uptime_seconds / total_seconds) * 100
    
    def aggregate(
        self,
        values: List[float],
        method: Aggregation,
    ) -> float:
        """값 집계"""
        if not values:
            return 0.0
            
        if method == Aggregation.SUM:
            return sum(values)
        elif method == Aggregation.AVG:
            return statistics.mean(values)
        elif method == Aggregation.MIN:
            return min(values)
        elif method == Aggregation.MAX:
            return max(values)
        elif method == Aggregation.COUNT:
            return float(len(values))
        elif method == Aggregation.P50:
            return self.calculate_latency_percentile(values, 0.50)
        elif method == Aggregation.P90:
            return self.calculate_latency_percentile(values, 0.90)
        elif method == Aggregation.P95:
            return self.calculate_latency_percentile(values, 0.95)
        elif method == Aggregation.P99:
            return self.calculate_latency_percentile(values, 0.99)
        else:
            return statistics.mean(values)
    
    def build_dashboard(
        self,
        period_hours: int = 24,
    ) -> KPIDashboard:
        """
        KPI 대시보드 생성
        
        DB에서 데이터를 조회하여 KPI 계산
        """
        now = datetime.utcnow()
        period_start = now - timedelta(hours=period_hours)
        
        dashboard = KPIDashboard(
            period_start=period_start,
            period_end=now,
        )
        
        # 기본 KPI 메트릭 추가 (DB 없이도 동작)
        # 실제 환경에서는 DB에서 조회한 데이터로 계산
        
        # 예시: 기본 메트릭
        dashboard.add_metric(KPIMetric(
            name="pipeline_throughput",
            kpi_type=KPIType.THROUGHPUT,
            value=0.0,
            unit="items/hour",
            threshold_warning=100,
            threshold_critical=50,
        ))
        
        dashboard.add_metric(KPIMetric(
            name="pipeline_error_rate",
            kpi_type=KPIType.ERROR_RATE,
            value=0.0,
            unit="%",
            threshold_warning=5.0,
            threshold_critical=10.0,
        ))
        
        dashboard.add_metric(KPIMetric(
            name="pipeline_success_rate",
            kpi_type=KPIType.SUCCESS_RATE,
            value=100.0,
            unit="%",
            threshold_warning=95.0,
            threshold_critical=90.0,
        ))
        
        dashboard.add_metric(KPIMetric(
            name="pipeline_p95_latency",
            kpi_type=KPIType.LATENCY,
            value=0.0,
            unit="seconds",
            threshold_warning=30.0,
            threshold_critical=60.0,
        ))
        
        dashboard.add_metric(KPIMetric(
            name="storage_cost",
            kpi_type=KPIType.COST,
            value=0.0,
            unit="USD/month",
            threshold_warning=100.0,
            threshold_critical=500.0,
        ))
        
        return dashboard
    
    def get_trend(
        self,
        metric_name: str,
        periods: int = 7,
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """메트릭 추세 분석"""
        now = datetime.utcnow()
        trend_data = []
        
        for i in range(periods):
            period_end = now - timedelta(hours=period_hours * i)
            period_start = period_end - timedelta(hours=period_hours)
            
            # 실제 환경에서는 DB에서 조회
            trend_data.append({
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "value": 0.0,  # placeholder
            })
        
        return {
            "metric_name": metric_name,
            "periods": periods,
            "period_hours": period_hours,
            "data": trend_data,
        }


class KPIRegistry:
    """KPI 정의 레지스트리"""
    
    # 표준 KPI 정의
    STANDARD_KPIS = {
        "pipeline_throughput": {
            "name": "Pipeline Throughput",
            "kpi_type": KPIType.THROUGHPUT,
            "unit": "items/hour",
            "description": "시간당 처리 아이템 수",
            "threshold_warning": 100,
            "threshold_critical": 50,
        },
        "pipeline_error_rate": {
            "name": "Pipeline Error Rate",
            "kpi_type": KPIType.ERROR_RATE,
            "unit": "%",
            "description": "전체 작업 대비 에러 비율",
            "threshold_warning": 5.0,
            "threshold_critical": 10.0,
        },
        "pipeline_success_rate": {
            "name": "Pipeline Success Rate",
            "kpi_type": KPIType.SUCCESS_RATE,
            "unit": "%",
            "description": "전체 작업 대비 성공 비율",
            "threshold_warning": 95.0,
            "threshold_critical": 90.0,
        },
        "pipeline_p50_latency": {
            "name": "Pipeline P50 Latency",
            "kpi_type": KPIType.LATENCY,
            "unit": "seconds",
            "description": "작업 완료 시간 50번째 백분위수",
            "threshold_warning": 10.0,
            "threshold_critical": 30.0,
        },
        "pipeline_p95_latency": {
            "name": "Pipeline P95 Latency",
            "kpi_type": KPIType.LATENCY,
            "unit": "seconds",
            "description": "작업 완료 시간 95번째 백분위수",
            "threshold_warning": 30.0,
            "threshold_critical": 60.0,
        },
        "pipeline_p99_latency": {
            "name": "Pipeline P99 Latency",
            "kpi_type": KPIType.LATENCY,
            "unit": "seconds",
            "description": "작업 완료 시간 99번째 백분위수",
            "threshold_warning": 60.0,
            "threshold_critical": 120.0,
        },
        "storage_usage": {
            "name": "Storage Usage",
            "kpi_type": KPIType.EFFICIENCY,
            "unit": "GB",
            "description": "총 스토리지 사용량",
            "threshold_warning": 100.0,
            "threshold_critical": 500.0,
        },
        "storage_cost_monthly": {
            "name": "Storage Cost (Monthly)",
            "kpi_type": KPIType.COST,
            "unit": "USD",
            "description": "월간 스토리지 비용 예상",
            "threshold_warning": 100.0,
            "threshold_critical": 500.0,
        },
        "queue_depth": {
            "name": "Queue Depth",
            "kpi_type": KPIType.THROUGHPUT,
            "unit": "items",
            "description": "현재 대기열 깊이",
            "threshold_warning": 1000,
            "threshold_critical": 5000,
        },
        "worker_utilization": {
            "name": "Worker Utilization",
            "kpi_type": KPIType.EFFICIENCY,
            "unit": "%",
            "description": "워커 사용률",
            "threshold_warning": 30.0,  # too low
            "threshold_critical": 10.0,
        },
    }
    
    @classmethod
    def get_kpi_definition(cls, kpi_id: str) -> Optional[Dict[str, Any]]:
        """KPI 정의 조회"""
        return cls.STANDARD_KPIS.get(kpi_id)
    
    @classmethod
    def list_kpis(cls) -> List[str]:
        """모든 KPI ID 목록"""
        return list(cls.STANDARD_KPIS.keys())


# 싱글톤 인스턴스
_kpi_calculator: Optional[KPICalculator] = None


def get_kpi_calculator(db_session=None) -> KPICalculator:
    """KPI 계산기 싱글톤 반환"""
    global _kpi_calculator
    if _kpi_calculator is None:
        _kpi_calculator = KPICalculator(db_session)
    return _kpi_calculator
