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
        **kwargs,
    ):
        self.target_count = target_count
        self.stage = stage
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.extra_config = kwargs

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

    def run_once(self):
        """파이프라인 1회 실행"""
        from mass_collector import MassCollector

        config = self._build_config()
        collector = MassCollector(config)

        if self.stage:
            collector.run(start_stage=self.stage, end_stage=self.stage)
        elif self.start_stage or self.end_stage:
            collector.run(
                start_stage=self.start_stage,
                end_stage=self.end_stage,
            )
        else:
            collector.run()

        # 일일 카운트 갱신
        self._check_date_reset()
        self._daily_count += config.target_count

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
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """CLI 파서 생성"""
    parser = argparse.ArgumentParser(
        description="P-ADE 메인 엔트리포인트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예:
  python main.py run-once                    # 단일 실행
  python main.py run-forever                 # 무한 루프
  python main.py run-forever --target 500    # 일일 목표 지정
  python main.py monitor-alerts              # 알림 모니터링
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="실행 모드")

    # run-once
    p_once = subparsers.add_parser("run-once", help="파이프라인 1회 실행")
    p_once.add_argument("--target", type=int, default=500)
    p_once.add_argument("--stage", help="단일 단계")
    p_once.add_argument("--start-stage", help="시작 단계")
    p_once.add_argument("--end-stage", help="종료 단계")

    # run-forever
    p_forever = subparsers.add_parser("run-forever", help="무한 루프 모드")
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

    if args.command == "run-forever":
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

    elif args.command == "run-once" or args.command is None:
        # 단일 실행 (기본)
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
