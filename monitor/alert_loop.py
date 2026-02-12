"""
알림 모니터링 루프 (Task 4.2)

주기적으로 시스템 메트릭을 수집하고 AlertManager를 통해 알림을 발생시킵니다.

Task 4.2 알림 조건:
- GPU 사용률 < 30%
- 큐 잔여 < 100개
- 실패율 > 40%
- 일일 목표 대비 < 40% (18시 기준)
"""

import os
import sys
import time
import signal
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================
# Task 4.2 알림 규칙 등록
# ============================================================

def register_task4_rules(manager):
    """
    Task 4.2 전용 알림 규칙 등록

    기존 register_default_rules()와 별개로 추가되는 규칙들:
    - gpu_util_low: GPU 사용률 30% 미만
    - queue_depleted: 큐 100개 미만
    - failure_rate_high: 실패율 40% 초과
    - target_behind: 18시 기준 목표 대비 40% 이하
    """
    from alerts.manager import AlertRule, AlertChannel
    from alerts.slack import AlertLevel

    # GPU 사용률 낮음
    manager.register_rule(AlertRule(
        name="gpu_util_low",
        condition=lambda gpu_util=100, **_: gpu_util < 30,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        severity=AlertLevel.WARNING,
        cooldown_seconds=600,
        description="GPU utilization below 30%",
        labels={"component": "gpu"},
    ))

    # 큐 고갈
    manager.register_rule(AlertRule(
        name="queue_depleted",
        condition=lambda queue_size=1000, **_: queue_size < 100,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        severity=AlertLevel.WARNING,
        cooldown_seconds=300,
        description="Queue nearly depleted (< 100 items)",
        labels={"component": "queue"},
    ))

    # 실패율 높음
    manager.register_rule(AlertRule(
        name="failure_rate_high",
        condition=lambda failure_rate=0, **_: failure_rate > 0.4,
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.LOG],
        severity=AlertLevel.ERROR,
        cooldown_seconds=900,
        description="Failure rate exceeded 40%",
        labels={"component": "pipeline"},
    ))

    # 일일 목표 지연 (18시 체크)
    manager.register_rule(AlertRule(
        name="target_behind",
        condition=lambda daily_progress=1.0, current_hour=0, **_: (
            current_hour == 18 and daily_progress < 0.4
        ),
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.LOG],
        severity=AlertLevel.ERROR,
        cooldown_seconds=3600,
        description="Behind daily target (< 40% at 18:00)",
        labels={"component": "pipeline"},
    ))

    logger.info("Task 4.2 알림 규칙 4개 등록 완료")


# ============================================================
# 메트릭 수집기
# ============================================================

class MetricsCollector:
    """
    시스템 메트릭 수집기

    GPU, 큐, 실패율, 일일 진행률 등을 수집하여
    AlertManager.check_rules()에 전달할 context dict를 생성합니다.
    """

    def __init__(
        self,
        daily_target: int = 500,
        db_path: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.daily_target = daily_target
        self.db_path = db_path or str(PROJECT_ROOT / "data" / "pade.db")
        self._redis = None
        self._redis_host = redis_host
        self._redis_port = redis_port

    def _get_redis(self):
        """Redis 연결 (lazy init)"""
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis(
                    host=self._redis_host,
                    port=self._redis_port,
                    decode_responses=True,
                )
                self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    def get_gpu_utilization(self) -> float:
        """GPU 사용률 조회 (%)"""
        # Redis에 캐시된 값 먼저 시도
        r = self._get_redis()
        if r:
            try:
                val = r.get("pade:gpu_util")
                if val is not None:
                    return float(val)
            except Exception:
                pass

        # 직접 nvidia-smi 호출
        try:
            from monitor.stats_collector import StatsCollector
            return StatsCollector.get_gpu_utilization()
        except Exception:
            return 0.0

    def get_queue_size(self) -> int:
        """큐 잔여 수 조회"""
        r = self._get_redis()
        if r:
            try:
                val = r.get("pade:queue_size")
                if val is not None:
                    return int(val)

                # 대안: 큐 리스트 길이
                val = r.llen("pade:download_queue")
                if val is not None:
                    return int(val)
            except Exception:
                pass

        # DB 폴백: pending 상태 비디오 수
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status = 'pending'"
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def get_failure_rate(self) -> float:
        """실패율 조회 (0.0 ~ 1.0)"""
        r = self._get_redis()
        if r:
            try:
                total = int(r.hget("pade:processing_stats", "total_processed") or 0)
                failed = int(r.hget("pade:processing_stats", "total_failed") or 0)
                if total > 0:
                    return failed / total
            except Exception:
                pass

        # DB 폴백
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status IN ('completed', 'failed')"
            )
            total = cursor.fetchone()[0]
            cursor = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status = 'failed'"
            )
            failed = cursor.fetchone()[0]
            conn.close()
            if total > 0:
                return failed / total
        except Exception:
            pass

        return 0.0

    def get_daily_progress(self) -> float:
        """일일 목표 대비 진행률 (0.0 ~ 1.0+)"""
        r = self._get_redis()
        if r:
            try:
                collected = int(r.get("pade:collected_today") or 0)
                if self.daily_target > 0:
                    return collected / self.daily_target
            except Exception:
                pass

        # DB 폴백: 오늘 완료된 비디오 수
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status = 'completed' "
                "AND date(created_at) = ?",
                (today,),
            )
            completed = cursor.fetchone()[0]
            conn.close()
            if self.daily_target > 0:
                return completed / self.daily_target
        except Exception:
            pass

        return 0.0

    def collect(self) -> Dict[str, Any]:
        """
        전체 메트릭 수집

        Returns:
            AlertManager.check_rules()에 전달할 context dict
        """
        context = {
            "gpu_util": self.get_gpu_utilization(),
            "queue_size": self.get_queue_size(),
            "failure_rate": self.get_failure_rate(),
            "daily_progress": self.get_daily_progress(),
            "current_hour": datetime.now().hour,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug(
            f"메트릭 수집: GPU={context['gpu_util']:.0f}%, "
            f"큐={context['queue_size']}, "
            f"실패율={context['failure_rate']:.1%}, "
            f"진행률={context['daily_progress']:.1%}"
        )

        return context


# ============================================================
# 알림 모니터링 루프
# ============================================================

class AlertMonitorLoop:
    """
    주기적 알림 체크 루프

    MetricsCollector로 시스템 메트릭을 수집하고
    AlertManager.check_rules()로 규칙을 평가합니다.
    """

    def __init__(
        self,
        alert_manager=None,
        metrics_collector: Optional[MetricsCollector] = None,
        check_interval: int = 300,
        daily_target: int = 500,
    ):
        from alerts.manager import get_alert_manager

        self.alert_manager = alert_manager or get_alert_manager()
        self.metrics = metrics_collector or MetricsCollector(
            daily_target=daily_target,
        )
        self.check_interval = check_interval
        self._shutdown = False

    def _handle_signal(self, signum, frame):
        logger.info(f"알림 모니터 종료 신호 수신: {signum}")
        self._shutdown = True

    def check_once(self) -> Dict[str, Any]:
        """1회 메트릭 수집 + 규칙 체크"""
        context = self.metrics.collect()
        self.alert_manager.check_rules(context)
        return context

    def run(self):
        """무한 루프 실행"""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            f"🔔 알림 모니터링 루프 시작 "
            f"(체크 간격: {self.check_interval}초)"
        )

        while not self._shutdown:
            try:
                self.check_once()
            except Exception as e:
                logger.error(f"알림 체크 실패: {e}", exc_info=True)

            # 종료 신호를 빠르게 감지하기 위해 잘게 쪼갠 sleep
            for _ in range(self.check_interval):
                if self._shutdown:
                    break
                time.sleep(1)

        logger.info("알림 모니터링 루프 종료")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="알림 모니터링 루프")
    parser.add_argument(
        "--interval", type=int, default=300, help="체크 간격(초)"
    )
    parser.add_argument(
        "--target", type=int, default=500, help="일일 수집 목표"
    )
    args = parser.parse_args()

    from alerts.manager import get_alert_manager, register_default_rules

    mgr = get_alert_manager()
    register_default_rules(mgr)
    register_task4_rules(mgr)

    loop = AlertMonitorLoop(
        alert_manager=mgr,
        check_interval=args.interval,
        daily_target=args.target,
    )
    loop.run()
