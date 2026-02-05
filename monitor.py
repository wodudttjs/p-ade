#!/usr/bin/env python
"""
CLI 모니터링 대시보드

MVP Phase 2 Week 8: Basic Dashboard (CLI 모니터링)
- CLI 기반 상태 모니터
- 처리 통계 출력
- 오류 로그 집계

GUI 대시보드(app.py)와 동일한 DataService 사용

사용법:
    python monitor.py                  # 실시간 모니터링
    python monitor.py --once           # 1회 출력
    python monitor.py --jobs           # 작업 목록
    python monitor.py --errors         # 오류 로그
    python monitor.py --watch          # 자동 새로고침 (5초)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정 (loguru 대신 기본 logging 사용)
import logging
logger = logging.getLogger(__name__)


# 터미널 색상 코드
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 형태로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes_size: float) -> str:
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"


def progress_bar(value: float, width: int = 20, color: str = Colors.GREEN) -> str:
    """진행률 바 생성"""
    filled = int(width * value)
    empty = width - filled
    return f"{color}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"


class CLIMonitor:
    """CLI 모니터링 클래스"""
    
    def __init__(self, db_url: Optional[str] = None):
        # DataService 직접 로드 (dashboard 패키지 전체 로드 방지)
        try:
            import importlib.util
            data_service_path = project_root / "dashboard" / "data_service.py"
            spec = importlib.util.spec_from_file_location("data_service", data_service_path)
            data_service_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_service_module)
            
            self.data_service = data_service_module.get_data_service(db_url)
            self.connected = self.data_service.is_connected()
        except Exception as e:
            logger.warning(f"DataService 초기화 실패: {e}")
            self.data_service = None
            self.connected = False
    
    def print_header(self, title: str = "P-ADE Monitor"):
        """헤더 출력"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        width = 60
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}🎬 {title}{Colors.RESET}")
        print(f"{Colors.DIM}{now}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*width}{Colors.RESET}\n")
    
    def print_connection_status(self):
        """연결 상태 출력"""
        if self.connected:
            print(f"{Colors.GREEN}✅ Database Connected{Colors.RESET}")
        else:
            print(f"{Colors.RED}❌ Database Not Connected (using mock data){Colors.RESET}")
        print()
    
    def print_kpi(self):
        """KPI 카드 출력"""
        if not self.data_service:
            print(f"{Colors.YELLOW}⚠ DataService not available{Colors.RESET}")
            return
        
        kpi = self.data_service.get_kpi()
        
        print(f"{Colors.BOLD}📊 KPI Overview{Colors.RESET}")
        print(f"{'─'*50}")
        
        # 비디오 통계
        print(f"  📹 Videos: {Colors.BOLD}{kpi.total_videos}{Colors.RESET} total, "
              f"{Colors.GREEN}{kpi.downloaded}{Colors.RESET} downloaded")
        
        # 에피소드 통계
        quality_rate = kpi.high_quality / max(kpi.episodes, 1)
        print(f"  🎬 Episodes: {Colors.BOLD}{kpi.episodes}{Colors.RESET} total, "
              f"{Colors.GREEN}{kpi.high_quality}{Colors.RESET} high quality "
              f"({quality_rate*100:.1f}%)")
        
        # 처리 통계
        success_color = Colors.GREEN if kpi.success_rate >= 0.9 else (
            Colors.YELLOW if kpi.success_rate >= 0.7 else Colors.RED
        )
        print(f"  ✅ Success Rate: {success_color}{kpi.success_rate*100:.1f}%{Colors.RESET} "
              f"{progress_bar(kpi.success_rate, 15, success_color)}")
        
        # 처리 시간
        print(f"  ⏱️ Avg Processing: {Colors.CYAN}{format_duration(kpi.avg_processing_time_sec)}{Colors.RESET}")
        
        # 큐 상태
        print(f"  📋 Queue: {Colors.YELLOW}{kpi.queue_depth}{Colors.RESET} pending, "
              f"{Colors.BLUE}{kpi.active_workers}{Colors.RESET} active")
        
        # 스토리지
        print(f"  💾 Storage: {Colors.MAGENTA}{kpi.storage_gb:.2f} GB{Colors.RESET} "
              f"(${kpi.monthly_cost:.2f}/month)")
        
        print()
    
    def print_job_stats(self):
        """작업 상태별 통계 출력"""
        if not self.data_service:
            return
        
        stats = self.data_service.get_job_stats()
        total = stats.get("total", 0)
        
        print(f"{Colors.BOLD}📋 Job Statistics{Colors.RESET}")
        print(f"{'─'*50}")
        
        status_config = [
            ("pending", "⏳", Colors.YELLOW),
            ("running", "🔄", Colors.BLUE),
            ("success", "✅", Colors.GREEN),
            ("fail", "❌", Colors.RED),
            ("skip", "⏭️", Colors.DIM),
        ]
        
        for status, icon, color in status_config:
            count = stats.get(status, 0)
            pct = count / max(total, 1) * 100
            bar = progress_bar(count / max(total, 1), 20, color)
            print(f"  {icon} {status.capitalize():10} {color}{count:5}{Colors.RESET} "
                  f"({pct:5.1f}%) {bar}")
        
        print(f"  {'─'*44}")
        print(f"  📊 {'Total':10} {Colors.BOLD}{total:5}{Colors.RESET}")
        print()
    
    def print_recent_jobs(self, limit: int = 10):
        """최근 작업 목록 출력"""
        if not self.data_service:
            return
        
        jobs = self.data_service.get_jobs(limit=limit)
        
        print(f"{Colors.BOLD}📋 Recent Jobs (Last {limit}){Colors.RESET}")
        print(f"{'─'*70}")
        
        # 헤더
        print(f"  {'Status':<8} {'Stage':<10} {'Video ID':<15} {'Duration':<10} {'Error':<20}")
        print(f"  {'─'*66}")
        
        status_icons = {
            "pending": f"{Colors.YELLOW}⏳{Colors.RESET}",
            "running": f"{Colors.BLUE}🔄{Colors.RESET}",
            "completed": f"{Colors.GREEN}✅{Colors.RESET}",
            "failed": f"{Colors.RED}❌{Colors.RESET}",
            "skipped": f"{Colors.DIM}⏭️{Colors.RESET}",
        }
        
        for job in jobs:
            icon = status_icons.get(job.status, "❓")
            duration = format_duration(job.duration_ms / 1000) if job.duration_ms else "—"
            error = (job.error_type or "")[:18] + ".." if job.error_type and len(job.error_type) > 20 else (job.error_type or "—")
            
            print(f"  {icon} {job.status:<6} {job.stage:<10} {job.video_id:<15} "
                  f"{duration:<10} {error:<20}")
        
        print()
    
    def print_errors(self, limit: int = 10):
        """오류 로그 출력"""
        if not self.data_service:
            return
        
        # 실패한 작업만 조회
        jobs = self.data_service.get_jobs(limit=limit, status="failed")
        
        print(f"{Colors.BOLD}{Colors.RED}❌ Recent Errors (Last {limit}){Colors.RESET}")
        print(f"{'─'*70}")
        
        if not jobs:
            print(f"  {Colors.GREEN}No errors found!{Colors.RESET}")
            print()
            return
        
        # 오류 유형별 집계
        error_counts = {}
        for job in jobs:
            error_type = job.error_type or "unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        print(f"\n  {Colors.BOLD}Error Summary:{Colors.RESET}")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {Colors.RED}•{Colors.RESET} {error_type}: {count}")
        
        print(f"\n  {Colors.BOLD}Details:{Colors.RESET}")
        for job in jobs[:5]:
            print(f"    {Colors.RED}✗{Colors.RESET} [{job.stage}] {job.video_id}")
            if job.log_snippet:
                print(f"      {Colors.DIM}{job.log_snippet[:60]}...{Colors.RESET}")
        
        print()
    
    def print_quality_stats(self):
        """품질 통계 출력"""
        if not self.data_service:
            return
        
        stats = self.data_service.get_quality_stats()
        
        print(f"{Colors.BOLD}📈 Quality Statistics{Colors.RESET}")
        print(f"{'─'*50}")
        
        # 통과율
        pass_color = Colors.GREEN if stats.pass_rate >= 0.8 else (
            Colors.YELLOW if stats.pass_rate >= 0.5 else Colors.RED
        )
        print(f"  Pass Rate: {pass_color}{stats.pass_rate*100:.1f}%{Colors.RESET} "
              f"({stats.passed}/{stats.passed + stats.failed})")
        print(f"  {progress_bar(stats.pass_rate, 30, pass_color)}")
        
        # 상세 통계
        print(f"\n  {Colors.BOLD}Metrics:{Colors.RESET}")
        print(f"    Confidence: {stats.confidence_mean:.3f} ± {stats.confidence_std:.3f}")
        print(f"    Jitter:     {stats.jitter_mean:.3f} ± {stats.jitter_std:.3f}")
        print(f"    Length:     {stats.length_mean:.1f} frames")
        
        print()
    
    def print_pipeline_status(self):
        """파이프라인 상태 출력"""
        print(f"{Colors.BOLD}🔄 Pipeline Status{Colors.RESET}")
        print(f"{'─'*50}")
        
        # 각 파일 존재 여부 확인
        data_dir = project_root / "data"
        
        stages = [
            ("raw", "Downloaded Videos", "*.mp4"),
            ("poses", "Extracted Poses", "*_pose.npz"),
            ("filtered", "Filtered Poses", "*.npz"),
            ("episodes", "Encoded Episodes", "*_episode.npz"),
        ]
        
        for folder, name, pattern in stages:
            path = data_dir / folder
            if path.exists():
                files = list(path.glob(pattern))
                count = len(files)
                size = sum(f.stat().st_size for f in files)
                status = f"{Colors.GREEN}✅{Colors.RESET}"
            else:
                count = 0
                size = 0
                status = f"{Colors.DIM}—{Colors.RESET}"
            
            print(f"  {status} {name:<20} {count:>4} files ({format_size(size)})")
        
        print()
    
    def print_full_dashboard(self, clear: bool = True):
        """전체 대시보드 출력"""
        if clear:
            clear_screen()
        self.print_header()
        self.print_connection_status()
        self.print_kpi()
        self.print_job_stats()
        self.print_pipeline_status()
        self.print_quality_stats()
    
    def watch(self, interval: int = 5):
        """실시간 모니터링"""
        print(f"{Colors.CYAN}Starting real-time monitoring (Ctrl+C to stop)...{Colors.RESET}")
        print(f"Refresh interval: {interval}s")
        time.sleep(1)
        
        try:
            while True:
                self.print_full_dashboard()
                print(f"{Colors.DIM}Next refresh in {interval}s... (Ctrl+C to stop){Colors.RESET}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped.{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="P-ADE CLI Monitor")
    
    parser.add_argument("--once", action="store_true", help="1회 출력 후 종료")
    parser.add_argument("--watch", action="store_true", help="실시간 모니터링")
    parser.add_argument("--interval", type=int, default=5, help="새로고침 간격 (초)")
    
    parser.add_argument("--jobs", action="store_true", help="작업 목록 출력")
    parser.add_argument("--errors", action="store_true", help="오류 로그 출력")
    parser.add_argument("--quality", action="store_true", help="품질 통계 출력")
    parser.add_argument("--pipeline", action="store_true", help="파이프라인 상태")
    parser.add_argument("--kpi", action="store_true", help="KPI만 출력")
    
    parser.add_argument("--limit", type=int, default=10, help="출력 개수 제한")
    parser.add_argument("--no-color", action="store_true", help="색상 비활성화")
    
    args = parser.parse_args()
    
    # 색상 비활성화
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")
    
    # 모니터 생성
    monitor = CLIMonitor()
    
    # 실행 모드
    if args.watch:
        monitor.watch(interval=args.interval)
    elif args.jobs:
        monitor.print_header("Job List")
        monitor.print_connection_status()
        monitor.print_recent_jobs(limit=args.limit)
    elif args.errors:
        monitor.print_header("Error Log")
        monitor.print_connection_status()
        monitor.print_errors(limit=args.limit)
    elif args.quality:
        monitor.print_header("Quality Stats")
        monitor.print_connection_status()
        monitor.print_quality_stats()
    elif args.pipeline:
        monitor.print_header("Pipeline Status")
        monitor.print_pipeline_status()
    elif args.kpi:
        monitor.print_header("KPI Dashboard")
        monitor.print_connection_status()
        monitor.print_kpi()
    else:
        # 기본: 전체 대시보드 (--once 이면 clear 안함)
        monitor.print_full_dashboard(clear=not args.once)
        
        if not args.once:
            print(f"{Colors.DIM}Use --watch for real-time monitoring{Colors.RESET}")


if __name__ == "__main__":
    main()
