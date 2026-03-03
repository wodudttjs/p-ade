#!/usr/bin/env python
"""
P-ADE 메인 엔트리포인트

systemd 서비스 및 CLI 통합 진입점입니다.

사용법:
    # 단일 실행 (기본)
    python main.py

    # 무한 루프 모드 (systemd용)
    python main.py run-forever

    # 무한 루프 + 커스텀 설정
    python main.py run-forever --target 500 --interval 60

    # 알림 모니터링만 실행
    python main.py monitor-alerts

    # 단일 실행 + 파이프라인 옵션
    python main.py run-once --target 200 --stage crawl
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Windows cp949 인코딩 문제 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================
# CollectionPipeline (MassCollector 래퍼)
# ============================================================

class CollectionPipeline:
    """
    MassCollector를 감싸는 파이프라인 래퍼.

    run-forever 모드에서 사용:
    - 일일 목표 달성 체크
    - 1회 실행 (run_once)
    - 리소스 정리 (cleanup)
    """

    def __init__(
        self,
        target_count: int = 500,
        stage: Optional[str] = None,
        start_stage: Optional[str] = None,
        end_stage: Optional[str] = None,
        dashboard_sync: bool = False,
        **kwargs,
    ):
        self.target_count = target_count
        self.stage = stage
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.extra_config = kwargs
        self.dashboard_sync = dashboard_sync

        # 일일 수집 카운트
        self._daily_count = 0
        self._daily_reset_date = datetime.now().date()

    def _build_config(self):
        """PipelineConfig 생성"""
        from mass_collector import PipelineConfig

        cfg_kwargs = {
            "target_count": self.target_count,
        }
        # extra_config에서 PipelineConfig 필드만 추출
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
        for k, v in self.extra_config.items():
            if k in valid_fields and v is not None:
                cfg_kwargs[k] = v

        return PipelineConfig(**cfg_kwargs)

    def _get_dashboard_callbacks(self):
        """대시보드 실시간 동기화 콜백 생성"""
        if not self.dashboard_sync:
            return {}, {}

        try:
            from dashboard.web_app import pipeline_state
        except ImportError:
            return {}, {}

        def on_stage_start(stage_name):
            pipeline_state["current_stage"] = stage_name
            pipeline_state["is_running"] = True

        def on_stage_complete(stage_name, result):
            pipeline_state["progress"][stage_name] = 100 if result.success else 0

        def on_log(message):
            pipeline_state["logs"].append(message)
            # 로그 크기 제한 (2000줄)
            if len(pipeline_state["logs"]) > 2000:
                pipeline_state["logs"] = pipeline_state["logs"][-1500:]

        return (
            {"on_stage_start": on_stage_start,
             "on_stage_complete": on_stage_complete,
             "on_log": on_log},
            pipeline_state,
        )

    def run_once(self):
        """파이프라인 1회 실행"""
        from mass_collector import MassCollector

        config = self._build_config()

        # 대시보드 콜백
        callbacks, p_state = self._get_dashboard_callbacks()

        collector = MassCollector(config, **callbacks)

        if p_state:
            p_state["is_running"] = True
            p_state["started_at"] = datetime.now().isoformat()
            p_state["iteration"] = p_state.get("iteration", 0) + 1
            # progress는 리셋하지 않고, 현재 반복의 시작을 표시
            p_state["logs"].append(
                f"\n{'━'*50}\n"
                f"[INFO] 🔄 반복 #{p_state['iteration']} 시작 ({datetime.now().strftime('%H:%M:%S')})"
            )

        try:
            if self.stage:
                collector.run(start_stage=self.stage, end_stage=self.stage)
            elif self.start_stage or self.end_stage:
                collector.run(
                    start_stage=self.start_stage,
                    end_stage=self.end_stage,
                )
            else:
                collector.run()
        finally:
            if p_state:
                # progress를 보존 (last_progress에 스냅샷)
                p_state["last_progress"] = dict(p_state.get("progress", {}))
                # 반복 히스토리에 기록
                iteration_record = {
                    "iteration": p_state.get("iteration", 0),
                    "started_at": p_state.get("started_at"),
                    "completed_at": datetime.now().isoformat(),
                    "progress": dict(p_state.get("progress", {})),
                }
                if "iteration_history" not in p_state:
                    p_state["iteration_history"] = []
                p_state["iteration_history"].append(iteration_record)
                # 히스토리 크기 제한 (최근 50개)
                if len(p_state["iteration_history"]) > 50:
                    p_state["iteration_history"] = p_state["iteration_history"][-50:]

                # jobs_history에도 기록
                try:
                    from dashboard.web_app import jobs_history, _save_jobs_history
                    completed_stages = sum(1 for v in p_state["progress"].values() if v >= 100)
                    total_stages = len(p_state["progress"])
                    job = {
                        "id": len(jobs_history) + 1,
                        "job_key": f"auto_{p_state.get('iteration', 0)}_{datetime.now().strftime('%H%M%S')}",
                        "stage": "all",
                        "status": "completed" if completed_stages == total_stages else "partial",
                        "started_at": p_state.get("started_at", ""),
                        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "progress": int(completed_stages / total_stages * 100) if total_stages > 0 else 0,
                        "stages_done": completed_stages,
                        "stages_total": total_stages,
                    }
                    jobs_history.insert(0, job)
                    _save_jobs_history()
                except ImportError:
                    pass

                p_state["is_running"] = False
                p_state["current_stage"] = None
                p_state["logs"].append(
                    f"[INFO] ✅ 반복 #{p_state.get('iteration', 0)} 완료 ({datetime.now().strftime('%H:%M:%S')})"
                )

        # 일일 카운트 갱신 — 실제 수집 수 기준
        self._check_date_reset()
        # 실제 다운로드된 파일 수 기반 카운트 (target_count 통째 더하지 않음)
        raw_dir = Path("data/raw")
        actual_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0
        self._daily_count = actual_count

        logger.info(
            f"파이프라인 1회 실행 완료 "
            f"(일일 누적: {self._daily_count}/{self.target_count})"
        )

    def is_daily_target_reached(self) -> bool:
        """일일 목표 달성 여부"""
        self._check_date_reset()
        return self._daily_count >= self.target_count

    def _check_date_reset(self):
        """자정 넘으면 카운트 초기화"""
        today = datetime.now().date()
        if today != self._daily_reset_date:
            logger.info(
                f"날짜 변경 감지: {self._daily_reset_date} → {today}, 카운트 초기화"
            )
            self._daily_count = 0
            self._daily_reset_date = today

    def cleanup(self):
        """리소스 정리"""
        logger.info("파이프라인 리소스 정리 중...")
        # 추후 DB 연결, 임시 파일 등 정리 대상 추가 가능


# ============================================================
# run-forever 모드
# ============================================================

_shutdown_requested = False


def _signal_handler(signum, frame):
    """시그널 핸들러 (SIGINT, SIGTERM)"""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info(f"종료 신호 수신: {sig_name}")
    _shutdown_requested = True


def wait_until_tomorrow(hour: int = 6):
    """
    다음 날 지정 시각까지 대기

    Args:
        hour: 재개 시각 (기본 06:00)
    """
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    target = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
    sleep_seconds = (target - now).total_seconds()

    if sleep_seconds <= 0:
        return

    logger.info(
        f"일일 목표 달성, {target.strftime('%Y-%m-%d %H:%M')}까지 "
        f"{sleep_seconds/3600:.1f}시간 대기"
    )

    # 60초 간격으로 체크 (종료 신호 대응)
    start = time.time()
    while time.time() - start < sleep_seconds:
        if _shutdown_requested:
            return
        time.sleep(min(60, sleep_seconds - (time.time() - start)))


def run_forever(
    target_count: int = 500,
    interval: int = 60,
    error_wait: int = 300,
    resume_hour: int = 6,
    **kwargs,
):
    """
    무한 루프 파이프라인

    Args:
        target_count: 일일 수집 목표
        interval: 반복 간 대기 시간(초)
        error_wait: 에러 발생 시 대기 시간(초)
        resume_hour: 목표 달성 후 재개 시각
    """
    global _shutdown_requested

    # 시그널 등록
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pipeline = CollectionPipeline(target_count=target_count, **kwargs)

    logger.info(
        f"🚀 run-forever 모드 시작 "
        f"(목표: {target_count}/일, 간격: {interval}초)"
    )

    iteration = 0
    while not _shutdown_requested:
        try:
            # 1. 일일 목표 달성 체크
            if pipeline.is_daily_target_reached():
                logger.info("일일 목표 달성. 내일까지 대기합니다.")
                wait_until_tomorrow(hour=resume_hour)
                continue

            # 2. 파이프라인 1회 실행
            iteration += 1
            logger.info(f"파이프라인 반복 #{iteration} 시작...")
            pipeline.run_once()

            # 3. 대기
            if not _shutdown_requested:
                logger.info(f"{interval}초 대기 중...")
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — 정상 종료합니다.")
            break

        except Exception as e:
            logger.error(f"파이프라인 에러: {e}", exc_info=True)
            if not _shutdown_requested:
                logger.info(f"{error_wait}초 대기 후 재시도...")
                time.sleep(error_wait)

    pipeline.cleanup()
    logger.info(f"run-forever 종료 (총 {iteration}회 반복)")


# ============================================================
# 알림 모니터 모드
# ============================================================

def run_alert_monitor(check_interval: int = 300):
    """
    알림 모니터링 루프 실행

    Args:
        check_interval: 체크 간격(초), 기본 5분
    """
    from alerts.manager import get_alert_manager, register_default_rules
    from monitor.alert_loop import AlertMonitorLoop, register_task4_rules

    manager = get_alert_manager()
    register_default_rules(manager)
    register_task4_rules(manager)

    loop = AlertMonitorLoop(alert_manager=manager, check_interval=check_interval)
    loop.run()


# ============================================================
# serve 모드: 대시보드 + 파이프라인 자동 실행
# ============================================================

def run_serve(
    target_count: int = 500,
    interval: int = 120,
    error_wait: int = 300,
    port: int = 5000,
    host: str = "0.0.0.0",
    **kwargs,
):
    """
    서버 모드: Flask 웹 대시보드를 띄우고, 백그라운드에서 파이프라인을 자동 반복 실행

    흐름: 크롤링 → 다운로드 → GPU 추출(YOLO+MediaPipe) → 품질 검사 → 업로드
    대시보드에서 실시간 모니터링 가능

    Args:
        target_count: 일일 수집 목표
        interval: 파이프라인 반복 간 대기(초)
        error_wait: 에러 발생 시 대기(초)
        port: 대시보드 포트
        host: 대시보드 호스트
    """
    global _shutdown_requested

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        f"🚀 P-ADE 서버 모드 시작\n"
        f"   대시보드: http://{host}:{port}\n"
        f"   파이프라인 목표: {target_count}/일, 간격: {interval}초"
    )

    # 1) 파이프라인 백그라운드 스레드 시작
    import threading

    def pipeline_loop():
        """백그라운드 파이프라인 반복 루프"""
        global _shutdown_requested

        # 잠시 대기 — Flask 서버가 먼저 시작되도록
        time.sleep(3)

        pipeline = CollectionPipeline(
            target_count=target_count,
            dashboard_sync=True,
            **kwargs,
        )
        iteration = 0

        logger.info("🔄 파이프라인 백그라운드 루프 시작")

        while not _shutdown_requested:
            try:
                iteration += 1
                logger.info(f"🔄 파이프라인 반복 #{iteration} 시작...")

                pipeline.run_once()

                if not _shutdown_requested:
                    wait_sec = 30
                    logger.info(f"⏳ {wait_sec}초 대기 중...")
                    try:
                        from dashboard.web_app import pipeline_state
                        pipeline_state["logs"].append(
                            f"[INFO] ⏳ {wait_sec}초 대기 후 다음 반복..."
                        )
                    except ImportError:
                        pass
                    time.sleep(wait_sec)

                    # progress를 다음 반복을 위해 리셋 (last_progress에 이미 보존됨)
                    try:
                        from dashboard.web_app import pipeline_state
                        for k in pipeline_state["progress"]:
                            pipeline_state["progress"][k] = 0
                    except ImportError:
                        pass

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"파이프라인 에러: {e}", exc_info=True)
                try:
                    from dashboard.web_app import pipeline_state
                    pipeline_state["is_running"] = False
                    pipeline_state["logs"].append(f"[ERROR] ❌ 파이프라인 에러: {e}")
                except ImportError:
                    pass
                if not _shutdown_requested:
                    logger.info(f"⏳ 30초 대기 후 재시도...")
                    time.sleep(30)

        logger.info(f"파이프라인 루프 종료 (총 {iteration}회)")

    pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipeline_thread.start()

    # 2) Flask 웹 대시보드 실행 (메인 스레드)
    try:
        from dashboard.web_app import app

        logger.info(f"🌐 웹 대시보드 시작: http://{host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"대시보드 시작 실패: {e}")
        # 대시보드 실패해도 파이프라인은 계속 실행
        logger.info("대시보드 없이 파이프라인만 실행합니다.")
        try:
            pipeline_thread.join()
        except KeyboardInterrupt:
            _shutdown_requested = True


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """CLI 파서 생성"""
    parser = argparse.ArgumentParser(
        description="P-ADE 메인 엔트리포인트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예:
  python main.py                             # 서버 모드 (대시보드 + 자동 파이프라인)
  python main.py serve                       # 위와 동일
  python main.py serve --target 500 --port 5000
  python main.py run-once                    # 단일 실행 (파이프라인 1회)
  python main.py run-forever                 # 무한 루프 (CLI만, 대시보드 없음)
  python main.py monitor-alerts              # 알림 모니터링
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="실행 모드")

    # serve (기본 모드 — 대시보드 + 파이프라인 자동)
    p_serve = subparsers.add_parser("serve", help="서버 모드 (대시보드 + 자동 파이프라인)")
    p_serve.add_argument("--target", type=int, default=500, help="일일 수집 목표")
    p_serve.add_argument("--interval", type=int, default=120, help="파이프라인 반복 간격(초)")
    p_serve.add_argument("--error-wait", type=int, default=300, help="에러 시 대기(초)")
    p_serve.add_argument("--port", type=int, default=5000, help="대시보드 포트")
    p_serve.add_argument("--host", default="0.0.0.0", help="대시보드 호스트")

    # run-once
    p_once = subparsers.add_parser("run-once", help="파이프라인 1회 실행")
    p_once.add_argument("--target", type=int, default=500)
    p_once.add_argument("--stage", help="단일 단계")
    p_once.add_argument("--start-stage", help="시작 단계")
    p_once.add_argument("--end-stage", help="종료 단계")

    # run-forever
    p_forever = subparsers.add_parser("run-forever", help="무한 루프 모드 (CLI만)")
    p_forever.add_argument("--target", type=int, default=500)
    p_forever.add_argument("--interval", type=int, default=60, help="반복 간 대기(초)")
    p_forever.add_argument("--error-wait", type=int, default=300, help="에러 시 대기(초)")
    p_forever.add_argument("--resume-hour", type=int, default=6, help="재개 시각 (0-23)")
    p_forever.add_argument("--stage", help="단일 단계")
    p_forever.add_argument("--start-stage", help="시작 단계")
    p_forever.add_argument("--end-stage", help="종료 단계")

    # monitor-alerts
    p_alert = subparsers.add_parser("monitor-alerts", help="알림 모니터링 루프")
    p_alert.add_argument("--interval", type=int, default=300, help="체크 간격(초)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        # 기본 모드: 서버 (대시보드 + 파이프라인 자동 실행)
        run_serve(
            target_count=getattr(args, "target", 500),
            interval=getattr(args, "interval", 120),
            error_wait=getattr(args, "error_wait", 300),
            port=getattr(args, "port", 5000),
            host=getattr(args, "host", "0.0.0.0"),
        )

    elif args.command == "run-forever":
        run_forever(
            target_count=args.target,
            interval=args.interval,
            error_wait=args.error_wait,
            resume_hour=args.resume_hour,
            stage=getattr(args, "stage", None),
            start_stage=getattr(args, "start_stage", None),
            end_stage=getattr(args, "end_stage", None),
        )

    elif args.command == "monitor-alerts":
        run_alert_monitor(check_interval=args.interval)

    elif args.command == "run-once":
        pipeline = CollectionPipeline(
            target_count=getattr(args, "target", 500),
            stage=getattr(args, "stage", None),
            start_stage=getattr(args, "start_stage", None),
            end_stage=getattr(args, "end_stage", None),
        )
        pipeline.run_once()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
