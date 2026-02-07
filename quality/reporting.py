"""
Quality Reporting

주간/월간 품질 보고서 생성
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import json

from core.logging_config import setup_logger
from quality.tests import QualityReport, TestResult
from quality.profiling import DatasetProfile

logger = setup_logger(__name__)


class ReportFormat(str, Enum):
    """보고서 형식"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


class ReportPeriod(str, Enum):
    """보고서 기간"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ReportSection:
    """보고서 섹션"""
    title: str
    content: Any
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "order": self.order,
        }


@dataclass
class QualityReportSummary:
    """품질 보고서 요약"""
    period: ReportPeriod
    start_date: datetime
    end_date: datetime
    total_datasets: int = 0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    success_rate: float = 0.0
    critical_issues: int = 0
    sections: List[ReportSection] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_section(self, title: str, content: Any, order: int = 0):
        self.sections.append(ReportSection(title, content, order))
        self.sections.sort(key=lambda s: s.order)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "summary": {
                "total_datasets": self.total_datasets,
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": self.success_rate,
                "critical_issues": self.critical_issues,
            },
            "sections": [s.to_dict() for s in self.sections],
            "generated_at": self.generated_at.isoformat(),
        }


class ReportGenerator:
    """
    품질 보고서 생성기
    
    Task 6.3.3: 주간/월간 품질 보고서
    """
    
    def __init__(self):
        self._test_results: List[QualityReport] = []
        self._profiles: List[DatasetProfile] = []
    
    def add_test_result(self, report: QualityReport):
        """테스트 결과 추가"""
        self._test_results.append(report)
    
    def add_profile(self, profile: DatasetProfile):
        """프로파일 추가"""
        self._profiles.append(profile)
    
    def generate_summary(
        self,
        period: ReportPeriod = ReportPeriod.WEEKLY,
        end_date: Optional[datetime] = None,
    ) -> QualityReportSummary:
        """보고서 요약 생성"""
        end_date = end_date or datetime.utcnow()
        
        if period == ReportPeriod.DAILY:
            start_date = end_date - timedelta(days=1)
        elif period == ReportPeriod.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        else:  # MONTHLY
            start_date = end_date - timedelta(days=30)
        
        # 기간 내 테스트 결과 필터링
        period_results = [
            r for r in self._test_results
            if r.started_at and start_date <= r.started_at <= end_date
        ]
        
        # 통계 계산
        total_tests = sum(r.total for r in period_results)
        passed_tests = sum(r.passed for r in period_results)
        failed_tests = sum(r.failed for r in period_results)
        critical_issues = sum(
            1 for r in period_results
            for t in r.results
            if t.result == TestResult.FAIL and t.severity.value == "critical"
        )
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 100.0
        
        summary = QualityReportSummary(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_datasets=len(period_results),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            critical_issues=critical_issues,
        )
        
        # 섹션 추가
        summary.add_section(
            "Test Results Overview",
            self._build_test_overview(period_results),
            order=1,
        )
        
        if failed_tests > 0:
            summary.add_section(
                "Failed Tests",
                self._build_failed_tests_section(period_results),
                order=2,
            )
        
        if critical_issues > 0:
            summary.add_section(
                "Critical Issues",
                self._build_critical_issues_section(period_results),
                order=3,
            )
        
        summary.add_section(
            "Dataset Statistics",
            self._build_dataset_stats_section(period_results),
            order=4,
        )
        
        return summary
    
    def _build_test_overview(self, results: List[QualityReport]) -> Dict[str, Any]:
        """테스트 개요 구성"""
        by_dataset = {}
        for r in results:
            by_dataset[r.dataset_name] = {
                "total": r.total,
                "passed": r.passed,
                "failed": r.failed,
                "success_rate": r.success_rate,
            }
        return by_dataset
    
    def _build_failed_tests_section(self, results: List[QualityReport]) -> List[Dict[str, Any]]:
        """실패한 테스트 섹션"""
        failed = []
        for r in results:
            for t in r.results:
                if t.result == TestResult.FAIL:
                    failed.append({
                        "dataset": r.dataset_name,
                        "test": t.test_name,
                        "severity": t.severity.value,
                        "message": t.message,
                        "timestamp": t.timestamp.isoformat(),
                    })
        return failed
    
    def _build_critical_issues_section(self, results: List[QualityReport]) -> List[Dict[str, Any]]:
        """크리티컬 이슈 섹션"""
        critical = []
        for r in results:
            for t in r.results:
                if t.result == TestResult.FAIL and t.severity.value == "critical":
                    critical.append({
                        "dataset": r.dataset_name,
                        "test": t.test_name,
                        "message": t.message,
                        "expected": t.expected,
                        "actual": t.actual,
                    })
        return critical
    
    def _build_dataset_stats_section(self, results: List[QualityReport]) -> Dict[str, Any]:
        """데이터셋 통계 섹션"""
        return {
            "datasets_tested": len(set(r.dataset_name for r in results)),
            "test_runs": len(results),
            "avg_tests_per_run": (
                sum(r.total for r in results) / len(results)
                if results else 0
            ),
        }
    
    def format_report(
        self,
        summary: QualityReportSummary,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """보고서 포맷팅"""
        if format == ReportFormat.JSON:
            return json.dumps(summary.to_dict(), indent=2, ensure_ascii=False)
        elif format == ReportFormat.HTML:
            return self._format_html(summary)
        elif format == ReportFormat.MARKDOWN:
            return self._format_markdown(summary)
        else:
            return self._format_text(summary)
    
    def _format_markdown(self, summary: QualityReportSummary) -> str:
        """마크다운 형식 보고서"""
        lines = [
            f"# Data Quality Report ({summary.period.value.title()})",
            "",
            f"**Period:** {summary.start_date.date()} ~ {summary.end_date.date()}",
            f"**Generated:** {summary.generated_at.isoformat()}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Datasets | {summary.total_datasets} |",
            f"| Total Tests | {summary.total_tests} |",
            f"| Passed | {summary.passed_tests} |",
            f"| Failed | {summary.failed_tests} |",
            f"| Success Rate | {summary.success_rate:.1f}% |",
            f"| Critical Issues | {summary.critical_issues} |",
            "",
        ]
        
        for section in summary.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            
            if isinstance(section.content, dict):
                for key, value in section.content.items():
                    lines.append(f"- **{key}:** {value}")
            elif isinstance(section.content, list):
                for item in section.content[:10]:  # 최대 10개
                    if isinstance(item, dict):
                        lines.append(f"- {item.get('test', item.get('dataset', 'Unknown'))}: {item.get('message', '')}")
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(str(section.content))
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_html(self, summary: QualityReportSummary) -> str:
        """HTML 형식 보고서"""
        status_color = "#28a745" if summary.success_rate >= 90 else (
            "#ffc107" if summary.success_rate >= 70 else "#dc3545"
        )
        
        sections_html = ""
        for section in summary.sections:
            content_html = ""
            if isinstance(section.content, dict):
                content_html = "<ul>" + "".join(
                    f"<li><strong>{k}:</strong> {v}</li>"
                    for k, v in section.content.items()
                ) + "</ul>"
            elif isinstance(section.content, list):
                content_html = "<ul>" + "".join(
                    f"<li>{item}</li>" for item in section.content[:10]
                ) + "</ul>"
            else:
                content_html = f"<p>{section.content}</p>"
            
            sections_html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                {content_html}
            </div>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: {status_color}; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .section {{ margin: 20px 0; }}
        .section h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>📊 Data Quality Report ({summary.period.value.title()})</h1>
    <p><strong>Period:</strong> {summary.start_date.date()} ~ {summary.end_date.date()}</p>
    
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary.total_datasets}</div>
            <div class="metric-label">Datasets</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.total_tests}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric">
            <div class="metric-value" style="color: #28a745">{summary.passed_tests}</div>
            <div class="metric-label">Passed</div>
        </div>
        <div class="metric">
            <div class="metric-value" style="color: #dc3545">{summary.failed_tests}</div>
            <div class="metric-label">Failed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
    </div>
    
    {sections_html}
    
    <footer style="margin-top: 40px; font-size: 12px; color: #666;">
        Generated at {summary.generated_at.isoformat()} by P-ADE Quality System
    </footer>
</body>
</html>
        """
    
    def _format_text(self, summary: QualityReportSummary) -> str:
        """텍스트 형식 보고서"""
        lines = [
            "=" * 60,
            f"DATA QUALITY REPORT ({summary.period.value.upper()})",
            "=" * 60,
            f"Period: {summary.start_date.date()} ~ {summary.end_date.date()}",
            "",
            "SUMMARY",
            "-" * 30,
            f"Total Datasets:   {summary.total_datasets}",
            f"Total Tests:      {summary.total_tests}",
            f"Passed:           {summary.passed_tests}",
            f"Failed:           {summary.failed_tests}",
            f"Success Rate:     {summary.success_rate:.1f}%",
            f"Critical Issues:  {summary.critical_issues}",
            "",
        ]
        
        for section in summary.sections:
            lines.append(section.title.upper())
            lines.append("-" * 30)
            
            if isinstance(section.content, dict):
                for key, value in section.content.items():
                    lines.append(f"  {key}: {value}")
            elif isinstance(section.content, list):
                for item in section.content[:10]:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"  {section.content}")
            
            lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"Generated: {summary.generated_at.isoformat()}")
        
        return "\n".join(lines)
    
    def clear(self):
        """데이터 초기화"""
        self._test_results.clear()
        self._profiles.clear()


# 싱글톤 인스턴스
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """보고서 생성기 싱글톤 반환"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


# 편의 함수
def generate_quality_report(
    period: ReportPeriod = ReportPeriod.WEEKLY,
    format: ReportFormat = ReportFormat.MARKDOWN,
) -> str:
    """품질 보고서 생성 편의 함수"""
    generator = get_report_generator()
    summary = generator.generate_summary(period)
    return generator.format_report(summary, format)
