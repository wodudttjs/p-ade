"""
시스템 통계 수집 워커

Redis에 실시간 통계를 저장하여 대시보드에서 조회할 수 있게 합니다.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class StatsCollector:
    """통계 수집기"""
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.r = None
        if REDIS_AVAILABLE:
            try:
                self.r = redis.Redis(host=host, port=port, decode_responses=True)
                self.r.ping()
            except:
                pass
        
        self._prev_values = {}
    
    def collect_crawl_stats(self):
        """크롤링 통계 수집"""
        if not self.r:
            return
        
        # 현재 카운트
        curr = int(self.r.hget("pade:crawl_stats", "total_completed") or 0)
        prev = self._prev_values.get("crawl_count", curr)
        
        # 분당 속도 (1초 간격 수집 → 60배)
        speed = (curr - prev) * 60
        self.r.set("pade:crawl_speed", speed)
        
        self._prev_values["crawl_count"] = curr
    
    def collect_download_stats(self):
        """다운로드 통계 수집"""
        if not self.r:
            return
        
        curr = int(self.r.get("pade:download_count") or 0)
        prev = self._prev_values.get("download_count", curr)
        
        speed = (curr - prev) * 60
        self.r.set("pade:download_speed", speed)
        
        self._prev_values["download_count"] = curr
    
    def collect_processing_stats(self):
        """GPU 처리 통계 수집"""
        if not self.r:
            return
        
        curr = int(self.r.hget("pade:processing_stats", "total_processed") or 0)
        prev = self._prev_values.get("process_count", curr)
        
        speed = (curr - prev) * 60
        self.r.set("pade:process_speed", speed)
        
        self._prev_values["process_count"] = curr
    
    def collect_gpu_stats(self):
        """GPU 사용률 수집"""
        if not self.r:
            return
        
        util = self.get_gpu_utilization()
        self.r.set("pade:gpu_util", util)
    
    def collect_quality_stats(self):
        """품질 통계 수집"""
        if not self.r:
            return
        
        # 품질 등급별 카운트는 evaluator에서 직접 업데이트
        pass
    
    def update_collected_today(self):
        """오늘 수집량 업데이트"""
        if not self.r:
            return
        
        # 처리 완료 + 품질 통과 수
        passed = int(self.r.hget("pade:quality_stats", "passed") or 0)
        self.r.set("pade:collected_today", passed)
    
    @staticmethod
    def get_gpu_utilization() -> float:
        """nvidia-smi로 GPU 사용률 조회"""
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ], stderr=subprocess.DEVNULL)
            return float(output.decode().strip().split('\n')[0])
        except:
            return 0.0
    
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """GPU 메모리 사용량 조회"""
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], stderr=subprocess.DEVNULL)
            used, total = output.decode().strip().split(',')
            return {"used": float(used), "total": float(total)}
        except:
            return {"used": 0, "total": 0}
    
    def run(self, interval_sec: float = 1.0):
        """수집 루프 실행"""
        print(f"📊 통계 수집 시작 (간격: {interval_sec}초)")
        
        if not self.r:
            print("❌ Redis 연결 실패")
            return
        
        while True:
            try:
                self.collect_crawl_stats()
                self.collect_download_stats()
                self.collect_processing_stats()
                self.collect_gpu_stats()
                self.update_collected_today()
                
            except Exception as e:
                print(f"수집 오류: {e}")
            
            time.sleep(interval_sec)


def publish_log(message: str, host: str = "localhost", port: int = 6379):
    """로그 메시지를 Redis pubsub에 발행"""
    if not REDIS_AVAILABLE:
        return
    
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.publish("pade:logs", message)
    except:
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="통계 수집 워커")
    parser.add_argument("--interval", type=float, default=1.0, help="수집 간격 (초)")
    
    args = parser.parse_args()
    
    collector = StatsCollector()
    collector.run(interval_sec=args.interval)
