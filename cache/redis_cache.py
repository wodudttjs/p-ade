"""
Redis 기반 크롤링 캐시 시스템

검색 결과 캐싱, 중복 비디오 필터링, 캐시 통계 기능을 제공합니다.
"""

import json
import hashlib
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheConfig:
    """캐시 설정"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    search_ttl: int = 21600  # 검색 결과 TTL: 6시간
    video_ttl: int = 604800  # 비디오 정보 TTL: 7일
    bloom_size: int = 10**9  # Bloom filter 비트 크기


class CrawlCache:
    """
    크롤링 캐시 시스템
    
    기능:
    - 검색 결과 캐싱 (TTL: 6시간)
    - 비디오 메타데이터 캐싱 (TTL: 7일)
    - 수집 완료 비디오 추적 (Bloom Filter 방식)
    - 캐시 통계
    
    사용법:
        cache = CrawlCache()
        
        # 검색 결과 캐시
        results = cache.get_search_results("robot arm")
        if not results:
            results = api_search("robot arm")
            cache.save_search_results("robot arm", results)
        
        # 중복 체크
        new_videos = [v for v in results if not cache.is_video_collected(v['id'])]
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._stats = {
            "search_hits": 0,
            "search_misses": 0,
            "video_hits": 0,
            "video_misses": 0,
            "bloom_checks": 0,
        }
        
        if REDIS_AVAILABLE:
            self._connect()
    
    def _connect(self) -> bool:
        """Redis 연결"""
        if not REDIS_AVAILABLE:
            return False
        
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # 연결 테스트
            self._client.ping()
            self._connected = True
            return True
        except redis.ConnectionError:
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False
    
    # =========================================================================
    # 검색 결과 캐싱
    # =========================================================================
    
    def get_search_results(self, keyword: str, source: str = "youtube") -> Optional[List[Dict]]:
        """
        검색 결과 캐시 조회
        
        Args:
            keyword: 검색 키워드
            source: 소스 플랫폼 (youtube, google_videos 등)
            
        Returns:
            캐시된 결과 리스트 또는 None (캐시 미스)
        """
        if not self._connected:
            return None
        
        key = f"search:{source}:{keyword.lower().strip()}"
        
        try:
            cached = self._client.get(key)
            if cached:
                self._stats["search_hits"] += 1
                return json.loads(cached)
            else:
                self._stats["search_misses"] += 1
                return None
        except Exception:
            return None
    
    def save_search_results(
        self, 
        keyword: str, 
        results: List[Dict], 
        source: str = "youtube",
        ttl: Optional[int] = None,
    ):
        """
        검색 결과 캐시 저장
        
        Args:
            keyword: 검색 키워드
            results: 검색 결과 리스트
            source: 소스 플랫폼
            ttl: TTL 초 (기본: 6시간)
        """
        if not self._connected or not results:
            return
        
        key = f"search:{source}:{keyword.lower().strip()}"
        ttl = ttl or self.config.search_ttl
        
        try:
            self._client.setex(key, ttl, json.dumps(results))
        except Exception:
            pass
    
    # =========================================================================
    # 비디오 메타데이터 캐싱
    # =========================================================================
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """비디오 메타데이터 캐시 조회"""
        if not self._connected:
            return None
        
        key = f"video:{video_id}"
        
        try:
            cached = self._client.get(key)
            if cached:
                self._stats["video_hits"] += 1
                return json.loads(cached)
            else:
                self._stats["video_misses"] += 1
                return None
        except Exception:
            return None
    
    def save_video_info(self, video_id: str, info: Dict, ttl: Optional[int] = None):
        """비디오 메타데이터 캐시 저장"""
        if not self._connected or not info:
            return
        
        key = f"video:{video_id}"
        ttl = ttl or self.config.video_ttl
        
        try:
            self._client.setex(key, ttl, json.dumps(info))
        except Exception:
            pass
    
    def get_videos_batch(self, video_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """배치 비디오 메타데이터 조회"""
        if not self._connected or not video_ids:
            return {}
        
        keys = [f"video:{vid}" for vid in video_ids]
        
        try:
            values = self._client.mget(keys)
            results = {}
            for vid, val in zip(video_ids, values):
                if val:
                    results[vid] = json.loads(val)
                    self._stats["video_hits"] += 1
                else:
                    results[vid] = None
                    self._stats["video_misses"] += 1
            return results
        except Exception:
            return {}
    
    # =========================================================================
    # 수집 완료 추적 (Bloom Filter 방식)
    # =========================================================================
    
    def _hash_video_id(self, video_id: str) -> int:
        """video_id를 해시하여 비트 위치 계산"""
        h = hashlib.sha256(video_id.encode()).hexdigest()
        return int(h, 16) % self.config.bloom_size
    
    def is_video_collected(self, video_id: str) -> bool:
        """
        비디오 수집 완료 여부 확인 (Bloom Filter)
        
        Note: False positive 가능 (이미 수집했다고 잘못 판단)
              False negative 없음 (수집 안 한 것은 정확히 판단)
        """
        if not self._connected:
            return False
        
        self._stats["bloom_checks"] += 1
        
        try:
            bit_pos = self._hash_video_id(video_id)
            return bool(self._client.getbit("collected_videos", bit_pos))
        except Exception:
            return False
    
    def mark_video_collected(self, video_id: str):
        """비디오 수집 완료 마킹"""
        if not self._connected:
            return
        
        try:
            bit_pos = self._hash_video_id(video_id)
            self._client.setbit("collected_videos", bit_pos, 1)
        except Exception:
            pass
    
    def mark_videos_collected(self, video_ids: List[str]):
        """여러 비디오 일괄 수집 완료 마킹"""
        if not self._connected or not video_ids:
            return
        
        try:
            pipe = self._client.pipeline()
            for vid in video_ids:
                bit_pos = self._hash_video_id(vid)
                pipe.setbit("collected_videos", bit_pos, 1)
            pipe.execute()
        except Exception:
            pass
    
    def filter_uncollected(self, video_ids: List[str]) -> List[str]:
        """수집되지 않은 비디오만 필터링"""
        return [vid for vid in video_ids if not self.is_video_collected(vid)]
    
    # =========================================================================
    # 통계 및 관리
    # =========================================================================
    
    @property
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        stats = self._stats.copy()
        
        # 히트율 계산
        total_search = stats["search_hits"] + stats["search_misses"]
        total_video = stats["video_hits"] + stats["video_misses"]
        
        stats["search_hit_rate"] = (
            stats["search_hits"] / total_search if total_search > 0 else 0
        )
        stats["video_hit_rate"] = (
            stats["video_hits"] / total_video if total_video > 0 else 0
        )
        stats["connected"] = self._connected
        
        return stats
    
    def clear_search_cache(self):
        """검색 결과 캐시 전체 삭제"""
        if not self._connected:
            return
        
        try:
            keys = self._client.keys("search:*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass
    
    def clear_video_cache(self):
        """비디오 메타데이터 캐시 전체 삭제"""
        if not self._connected:
            return
        
        try:
            keys = self._client.keys("video:*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass
    
    def clear_bloom_filter(self):
        """Bloom filter 초기화"""
        if not self._connected:
            return
        
        try:
            self._client.delete("collected_videos")
        except Exception:
            pass
    
    def get_cache_size(self) -> Dict[str, int]:
        """캐시 크기 정보"""
        if not self._connected:
            return {"search_keys": 0, "video_keys": 0}
        
        try:
            search_keys = len(self._client.keys("search:*"))
            video_keys = len(self._client.keys("video:*"))
            return {"search_keys": search_keys, "video_keys": video_keys}
        except Exception:
            return {"search_keys": 0, "video_keys": 0}


# 싱글톤 인스턴스 (옵션)
_default_cache: Optional[CrawlCache] = None


def get_cache() -> CrawlCache:
    """기본 캐시 인스턴스 반환"""
    global _default_cache
    if _default_cache is None:
        _default_cache = CrawlCache()
    return _default_cache


class CacheMonitor:
    """
    캐시 통계 실시간 모니터링
    
    주기적으로 캐시 상태를 수집하고, 히트율/메모리 사용량 등을 추적합니다.
    
    사용법:
        monitor = CacheMonitor()
        stats = monitor.get_realtime_stats()
        report = monitor.generate_report()
    """
    
    def __init__(self, cache: Optional[CrawlCache] = None):
        self._cache = cache or get_cache()
        self._history: List[Dict] = []
        self._max_history = 100
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """실시간 캐시 통계 조회"""
        if not self._cache or not self._cache.is_connected:
            return {
                "connected": False,
                "message": "Redis 미연결",
            }
        
        stats = self._cache.stats
        cache_size = self._cache.get_cache_size()
        
        # Redis 서버 정보
        server_info = {}
        try:
            info = self._cache._client.info("memory")
            server_info = {
                "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "used_memory_peak_mb": round(info.get("used_memory_peak", 0) / (1024 * 1024), 2),
                "total_keys": self._cache._client.dbsize(),
            }
        except Exception:
            pass
        
        result = {
            "connected": True,
            "search_hits": stats["search_hits"],
            "search_misses": stats["search_misses"],
            "search_hit_rate": round(stats["search_hit_rate"] * 100, 1),
            "video_hits": stats["video_hits"],
            "video_misses": stats["video_misses"],
            "video_hit_rate": round(stats["video_hit_rate"] * 100, 1),
            "bloom_checks": stats["bloom_checks"],
            "search_keys": cache_size.get("search_keys", 0),
            "video_keys": cache_size.get("video_keys", 0),
            "timestamp": datetime.now().isoformat(),
            **server_info,
        }
        
        # 히스토리에 기록
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return result
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """최근 통계 히스토리"""
        return self._history[-limit:]
    
    def generate_report(self) -> str:
        """캐시 상태 보고서 생성"""
        stats = self.get_realtime_stats()
        
        if not stats.get("connected"):
            return "❌ Redis 미연결"
        
        lines = [
            "📊 캐시 모니터링 보고서",
            "=" * 40,
            "",
            "🔍 검색 캐시:",
            f"  히트: {stats['search_hits']}회",
            f"  미스: {stats['search_misses']}회",
            f"  히트율: {stats['search_hit_rate']}%",
            f"  저장된 키: {stats['search_keys']}개",
            "",
            "📹 비디오 캐시:",
            f"  히트: {stats['video_hits']}회",
            f"  미스: {stats['video_misses']}회",
            f"  히트율: {stats['video_hit_rate']}%",
            f"  저장된 키: {stats['video_keys']}개",
            "",
            "🔖 Bloom Filter:",
            f"  체크 횟수: {stats['bloom_checks']}회",
        ]
        
        if "used_memory_mb" in stats:
            lines.extend([
                "",
                "💾 Redis 메모리:",
                f"  현재: {stats['used_memory_mb']} MB",
                f"  피크: {stats['used_memory_peak_mb']} MB",
                f"  전체 키: {stats['total_keys']}개",
            ])
        
        return "\n".join(lines)


# 싱글톤 모니터 인스턴스
_default_monitor: Optional[CacheMonitor] = None


def get_monitor() -> CacheMonitor:
    """기본 모니터 인스턴스 반환"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = CacheMonitor()
    return _default_monitor
