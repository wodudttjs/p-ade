"""
Grafana Dashboard Configuration

Grafana 대시보드 JSON 생성기
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class PanelType(str, Enum):
    """패널 유형"""
    GRAPH = "graph"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    LOGS = "logs"
    ALERT_LIST = "alertlist"
    TEXT = "text"
    TIMESERIES = "timeseries"
    BARGAUGE = "bargauge"
    PIECHART = "piechart"


class DataSource(str, Enum):
    """데이터 소스"""
    PROMETHEUS = "prometheus"
    LOKI = "loki"
    POSTGRES = "postgres"


@dataclass
class GrafanaPanel:
    """Grafana 패널 정의"""
    title: str
    panel_type: PanelType
    queries: List[Dict[str, Any]]
    grid_pos: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})
    description: str = ""
    datasource: str = "prometheus"
    unit: str = ""
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        panel = {
            "title": self.title,
            "type": self.panel_type.value,
            "datasource": {"type": self.datasource, "uid": f"${{{self.datasource.upper()}}}"},
            "gridPos": self.grid_pos,
            "description": self.description,
            "targets": self.queries,
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                },
            },
            "options": self.options,
        }
        
        if self.thresholds:
            panel["fieldConfig"]["defaults"]["thresholds"] = {
                "mode": "absolute",
                "steps": self.thresholds,
            }
            
        return panel


@dataclass
class GrafanaRow:
    """Grafana 행 정의"""
    title: str
    panels: List[GrafanaPanel] = field(default_factory=list)
    collapsed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "type": "row",
            "collapsed": self.collapsed,
            "panels": [p.to_dict() for p in self.panels],
        }


@dataclass
class GrafanaDashboard:
    """Grafana 대시보드 정의"""
    title: str
    uid: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    rows: List[GrafanaRow] = field(default_factory=list)
    panels: List[GrafanaPanel] = field(default_factory=list)
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    variables: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        all_panels = []
        panel_id = 1
        y_pos = 0
        
        # 개별 패널 추가
        for panel in self.panels:
            p = panel.to_dict()
            p["id"] = panel_id
            p["gridPos"]["y"] = y_pos
            all_panels.append(p)
            panel_id += 1
            y_pos += panel.grid_pos["h"]
        
        # 행 추가
        for row in self.rows:
            row_dict = {
                "id": panel_id,
                "title": row.title,
                "type": "row",
                "collapsed": row.collapsed,
                "gridPos": {"x": 0, "y": y_pos, "w": 24, "h": 1},
                "panels": [],
            }
            all_panels.append(row_dict)
            panel_id += 1
            y_pos += 1
            
            for panel in row.panels:
                p = panel.to_dict()
                p["id"] = panel_id
                p["gridPos"]["y"] = y_pos
                all_panels.append(p)
                panel_id += 1
            
            if row.panels:
                y_pos += max(p.grid_pos["h"] for p in row.panels)
        
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "timezone": "browser",
            "schemaVersion": 38,
            "version": 1,
            "refresh": self.refresh,
            "time": {
                "from": self.time_from,
                "to": self.time_to,
            },
            "templating": {
                "list": self.variables,
            },
            "panels": all_panels,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class DashboardBuilder:
    """
    Grafana 대시보드 빌더
    
    Task 6.1.4: Grafana 대시보드 설정
    """
    
    @staticmethod
    def create_prometheus_query(
        expr: str,
        legend: str = "{{instance}}",
        ref_id: str = "A",
    ) -> Dict[str, Any]:
        """Prometheus 쿼리 생성"""
        return {
            "expr": expr,
            "legendFormat": legend,
            "refId": ref_id,
            "datasource": {"type": "prometheus", "uid": "${PROMETHEUS}"},
        }
    
    @staticmethod
    def create_stat_thresholds(
        green: float = 0,
        yellow: float = 70,
        red: float = 90,
    ) -> List[Dict[str, Any]]:
        """Stat 패널 임계값 생성"""
        return [
            {"color": "green", "value": green},
            {"color": "yellow", "value": yellow},
            {"color": "red", "value": red},
        ]
    
    def build_overview_dashboard(self) -> GrafanaDashboard:
        """전체 현황 대시보드 생성"""
        dashboard = GrafanaDashboard(
            title="P-ADE Pipeline Overview",
            uid="p-ade-overview",
            description="파이프라인 전체 현황 대시보드",
            tags=["p-ade", "pipeline", "overview"],
            variables=[
                {
                    "name": "PROMETHEUS",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {"text": "Prometheus", "value": "prometheus"},
                },
            ],
        )
        
        # 요약 메트릭 행
        summary_row = GrafanaRow(title="Summary")
        
        # 총 처리량
        summary_row.panels.append(GrafanaPanel(
            title="Total Jobs (24h)",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'sum(increase(pipeline_jobs_total[24h]))',
                legend="Total",
            )],
            grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
            unit="short",
        ))
        
        # 성공률
        summary_row.panels.append(GrafanaPanel(
            title="Success Rate",
            panel_type=PanelType.GAUGE,
            queries=[self.create_prometheus_query(
                'sum(rate(pipeline_jobs_total{status="success"}[1h])) / sum(rate(pipeline_jobs_total[1h])) * 100',
                legend="Rate",
            )],
            grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
            unit="percent",
            thresholds=self.create_stat_thresholds(90, 95, 99),
        ))
        
        # 에러 수
        summary_row.panels.append(GrafanaPanel(
            title="Errors (1h)",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'sum(increase(pipeline_errors_total[1h]))',
                legend="Errors",
            )],
            grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
            unit="short",
            thresholds=[
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 10},
                {"color": "red", "value": 50},
            ],
        ))
        
        # 대기열 깊이
        summary_row.panels.append(GrafanaPanel(
            title="Queue Depth",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'sum(pipeline_queue_depth)',
                legend="Pending",
            )],
            grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
            unit="short",
        ))
        
        # 평균 처리 시간
        summary_row.panels.append(GrafanaPanel(
            title="Avg Duration (P95)",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'histogram_quantile(0.95, sum(rate(pipeline_job_duration_seconds_bucket[5m])) by (le))',
                legend="P95",
            )],
            grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
            unit="s",
        ))
        
        # 인플라이트 작업
        summary_row.panels.append(GrafanaPanel(
            title="In-Flight Jobs",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'sum(pipeline_inflight_jobs)',
                legend="Active",
            )],
            grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
            unit="short",
        ))
        
        dashboard.rows.append(summary_row)
        
        # 처리량 그래프 행
        throughput_row = GrafanaRow(title="Throughput")
        
        throughput_row.panels.append(GrafanaPanel(
            title="Jobs per Stage",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'sum by (stage) (rate(pipeline_jobs_total[5m]))',
                legend="{{stage}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            unit="ops",
        ))
        
        throughput_row.panels.append(GrafanaPanel(
            title="Items Processed",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'sum by (item_type) (rate(pipeline_items_processed_total[5m]))',
                legend="{{item_type}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="short",
        ))
        
        dashboard.rows.append(throughput_row)
        
        # 지연시간 그래프 행
        latency_row = GrafanaRow(title="Latency")
        
        latency_row.panels.append(GrafanaPanel(
            title="Job Duration by Stage (P95)",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'histogram_quantile(0.95, sum by (stage, le) (rate(pipeline_job_duration_seconds_bucket[5m])))',
                legend="{{stage}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            unit="s",
        ))
        
        latency_row.panels.append(GrafanaPanel(
            title="Queue Latency (P95)",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'histogram_quantile(0.95, sum by (stage, le) (rate(pipeline_queue_latency_seconds_bucket[5m])))',
                legend="{{stage}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="s",
        ))
        
        dashboard.rows.append(latency_row)
        
        # 에러 그래프 행
        error_row = GrafanaRow(title="Errors")
        
        error_row.panels.append(GrafanaPanel(
            title="Errors by Type",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'sum by (error_type) (rate(pipeline_errors_total[5m]))',
                legend="{{error_type}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            unit="short",
        ))
        
        error_row.panels.append(GrafanaPanel(
            title="Errors by Stage",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'sum by (stage) (rate(pipeline_errors_total[5m]))',
                legend="{{stage}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="short",
        ))
        
        dashboard.rows.append(error_row)
        
        # 리소스 그래프 행
        resource_row = GrafanaRow(title="Resources")
        
        resource_row.panels.append(GrafanaPanel(
            title="CPU Usage",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'pipeline_cpu_percent',
                legend="CPU",
            )],
            grid_pos={"x": 0, "y": 0, "w": 8, "h": 6},
            unit="percent",
        ))
        
        resource_row.panels.append(GrafanaPanel(
            title="Memory Usage",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'pipeline_memory_bytes',
                legend="Memory",
            )],
            grid_pos={"x": 8, "y": 0, "w": 8, "h": 6},
            unit="bytes",
        ))
        
        resource_row.panels.append(GrafanaPanel(
            title="GPU Utilization",
            panel_type=PanelType.TIMESERIES,
            queries=[self.create_prometheus_query(
                'pipeline_gpu_util_percent',
                legend="GPU {{gpu_id}}",
            )],
            grid_pos={"x": 16, "y": 0, "w": 8, "h": 6},
            unit="percent",
        ))
        
        dashboard.rows.append(resource_row)
        
        # 스토리지 행
        storage_row = GrafanaRow(title="Storage")
        
        storage_row.panels.append(GrafanaPanel(
            title="Storage by Provider",
            panel_type=PanelType.PIECHART,
            queries=[self.create_prometheus_query(
                'sum by (provider) (pipeline_storage_bytes)',
                legend="{{provider}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 8, "h": 6},
            unit="bytes",
        ))
        
        storage_row.panels.append(GrafanaPanel(
            title="Storage by Class",
            panel_type=PanelType.PIECHART,
            queries=[self.create_prometheus_query(
                'sum by (storage_class) (pipeline_storage_bytes)',
                legend="{{storage_class}}",
            )],
            grid_pos={"x": 8, "y": 0, "w": 8, "h": 6},
            unit="bytes",
        ))
        
        storage_row.panels.append(GrafanaPanel(
            title="Monthly Cost Estimate",
            panel_type=PanelType.STAT,
            queries=[self.create_prometheus_query(
                'sum(pipeline_storage_cost_usd_monthly_estimate)',
                legend="Cost",
            )],
            grid_pos={"x": 16, "y": 0, "w": 8, "h": 6},
            unit="currencyUSD",
        ))
        
        dashboard.rows.append(storage_row)
        
        return dashboard
    
    def build_alerts_dashboard(self) -> GrafanaDashboard:
        """알림 대시보드 생성"""
        dashboard = GrafanaDashboard(
            title="P-ADE Alerts",
            uid="p-ade-alerts",
            description="파이프라인 알림 대시보드",
            tags=["p-ade", "alerts"],
        )
        
        # 알림 목록 패널
        dashboard.panels.append(GrafanaPanel(
            title="Active Alerts",
            panel_type=PanelType.ALERT_LIST,
            queries=[],
            grid_pos={"x": 0, "y": 0, "w": 24, "h": 10},
            options={
                "showOptions": "current",
                "stateFilter": {"ok": False, "paused": False, "alerting": True, "pending": True},
            },
        ))
        
        return dashboard
    
    def export_all_dashboards(self) -> Dict[str, str]:
        """모든 대시보드 JSON 내보내기"""
        dashboards = {
            "overview": self.build_overview_dashboard().to_json(),
            "alerts": self.build_alerts_dashboard().to_json(),
        }
        return dashboards


# Convenience function
def generate_grafana_dashboards() -> Dict[str, str]:
    """Grafana 대시보드 JSON 생성"""
    builder = DashboardBuilder()
    return builder.export_all_dashboards()
