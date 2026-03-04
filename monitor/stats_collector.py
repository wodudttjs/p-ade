"""
시스템 통계 수집 워커

Redis에 실시간 통계를 저장하여 대시보드에서 조회할 수 있게 합니다.
"""

import sys
import time
import shutil
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
    """통계 수집기 (D3-2: 메트릭 수집 강화)"""
    
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

    # ─── D3-2: 신규 메트릭 5개 ────────────────────────────────────────────────

    def collect_disk_usage(self):
        """디스크 사용량 수집 (1분마다)"""
        if not self.r:
            return
        try:
            raw_dir = PROJECT_ROOT / "data" / "raw"
            episodes_dir = PROJECT_ROOT / "data" / "episodes"

            usage = shutil.disk_usage(str(PROJECT_ROOT))
            gb = 1024 ** 3

            raw_size = sum(f.stat().st_size for f in raw_dir.glob("*") if f.is_file()) if raw_dir.exists() else 0
            ep_size = sum(f.stat().st_size for f in episodes_dir.glob("*") if f.is_file()) if episodes_dir.exists() else 0

            pipe = self.r.pipeline()
            pipe.hset("pade:stats:disk", "total_gb", f"{usage.total / gb:.1f}")
            pipe.hset("pade:stats:disk", "used_gb", f"{usage.used / gb:.1f}")
            pipe.hset("pade:stats:disk", "free_gb", f"{usage.free / gb:.1f}")
            pipe.hset("pade:stats:disk", "raw_gb", f"{raw_size / gb:.2f}")
            pipe.hset("pade:stats:disk", "episodes_gb", f"{ep_size / gb:.2f}")
            pipe.expire("pade:stats:disk", 86400)
            pipe.execute()
        except Exception:
            pass

    def collect_download_speed(self):
        """다운로드 속도 수집 (10초마다, 5분 이동 평균)"""
        if not self.r:
            return
        try:
            curr = int(self.r.get("pade:download_count") or 0)
            prev = self._prev_values.get("dl_speed_prev", curr)
            speed = (curr - prev) * 6  # 10초 간격 → 분당
            self._prev_values["dl_speed_prev"] = curr

            # 시계열 저장 (LPUSH + LTRIM)
            self.r.lpush("pade:stats:download_speed", speed)
            self.r.ltrim("pade:stats:download_speed", 0, 29)  # 최근 30개 (5분)
            self.r.expire("pade:stats:download_speed", 86400)
        except Exception:
            pass

    def collect_duplicate_rate(self):
        """중복률 수집 (1분마다)"""
        if not self.r:
            return
        try:
            total = int(self.r.hget("pade:crawl_stats", "total_completed") or 0)
            dupes = int(self.r.hget("pade:crawl_stats", "total_duplicates") or 0)
            rate = (dupes / total * 100) if total > 0 else 0.0

            self.r.lpush("pade:stats:duplicate_rate", f"{rate:.1f}")
            self.r.ltrim("pade:stats:duplicate_rate", 0, 59)  # 최근 60개 (1시간)
            self.r.expire("pade:stats:duplicate_rate", 86400)
        except Exception:
            pass

    def collect_gpu_memory(self):
        """GPU별 VRAM 수집 (10초마다)"""
        if not self.r:
            return
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], stderr=subprocess.DEVNULL)

            pipe = self.r.pipeline()
            for line in output.decode().strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    idx, used, total, util = parts[0], parts[1], parts[2], parts[3]
                    pipe.hset("pade:stats:gpu", f"gpu{idx}_vram_used_mb", used)
                    pipe.hset("pade:stats:gpu", f"gpu{idx}_vram_total_mb", total)
                    pipe.hset("pade:stats:gpu", f"gpu{idx}_util_percent", util)
            pipe.expire("pade:stats:gpu", 86400)
            pipe.execute()
        except Exception:
            pass

    def collect_queue_depth(self):
        """큐 깊이 수집 (30초마다)"""
        if not self.r:
            return
        try:
            pipe = self.r.pipeline()

            # 크롤링 큐
            crawl_q = self.r.llen("pade:queue:crawl") or 0
            pipe.hset("pade:stats:queues", "crawl", crawl_q)

            # 다운로드 큐
            dl_q = self.r.llen("pade:queue:download") or self.r.llen("pade:download_queue") or 0
            pipe.hset("pade:stats:queues", "download", dl_q)

            # 처리 큐
            proc_q = self.r.llen("pade:queue:process") or 0
            pipe.hset("pade:stats:queues", "process", proc_q)

            pipe.expire("pade:stats:queues", 86400)
            pipe.execute()
        except Exception:
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
        """수집 루프 실행 (D3-2: 차등 간격 스케줄링)"""
        print(f"📊 통계 수집 시작 (기본 간격: {interval_sec}초)")
        
        if not self.r:
            print("❌ Redis 연결 실패")
            return
        
        tick = 0
        while True:
            try:
                # 매 틱 (1초): 기존 수집
                self.collect_crawl_stats()
                self.collect_download_stats()
                self.collect_processing_stats()
                self.collect_gpu_stats()
                self.update_collected_today()
                
                # 10초마다: download_speed, gpu_memory
                if tick % 10 == 0:
                    self.collect_download_speed()
                    self.collect_gpu_memory()
                
                # 30초마다: queue_depth
                if tick % 30 == 0:
                    self.collect_queue_depth()
                
                # 60초마다: disk_usage, duplicate_rate
                if tick % 60 == 0:
                    self.collect_disk_usage()
                    self.collect_duplicate_rate()
                
                tick += 1
                
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
