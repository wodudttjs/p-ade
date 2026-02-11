"""
GPU 3-Stream 병렬 처리 매니저

CUDA Stream을 활용한 3개 영상 동시 처리를 구현합니다.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)

# PyTorch/CUDA 임포트
try:
    import torch
    import torch.cuda as cuda
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    torch = None


@dataclass
class StreamConfig:
    """스트림 설정"""
    num_streams: int = 3              # 동시 스트림 수
    vram_limit_gb: float = 9.0        # VRAM 한계 (GB)
    target_fps: int = 30              # 기본 타겟 FPS
    long_video_fps: int = 15          # 긴 영상(60초+) FPS
    long_video_threshold_sec: int = 60


class GPU3StreamManager:
    """
    GPU 3-Stream 병렬 처리 매니저
    
    CUDA Stream을 사용하여 3개의 영상을 동시에 처리합니다.
    
    사용법:
        manager = GPU3StreamManager()
        results = manager.process_batch(video_paths)
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._lock = threading.Lock()
        
        # 통계
        self._stats = {
            "total_processed": 0,
            "total_time_sec": 0,
            "peak_vram_gb": 0,
        }
        
        # CUDA 스트림 초기화
        self._streams: List = []
        self._device = None
        
        if CUDA_AVAILABLE:
            self._init_cuda()
        else:
            logger.warning("⚠️ CUDA 사용 불가 - CPU 모드로 동작")
    
    def _init_cuda(self):
        """CUDA 초기화"""
        self._device = torch.device("cuda:0")
        self._streams = [cuda.Stream() for _ in range(self.config.num_streams)]
        
        # GPU 정보 로깅
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU 초기화: {gpu_name} ({total_memory:.1f} GB)")
    
    # =========================================================================
    # VRAM 모니터링
    # =========================================================================
    
    def get_vram_usage(self) -> Dict[str, float]:
        """VRAM 사용량 조회 (GB)"""
        if not CUDA_AVAILABLE:
            return {"allocated": 0, "reserved": 0, "available": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "available": total - reserved,
            "total": total,
        }
    
    def check_vram_health(self) -> bool:
        """VRAM 상태 확인"""
        usage = self.get_vram_usage()
        
        if usage["allocated"] > self.config.vram_limit_gb:
            logger.warning(f"⚠️ VRAM 한계 초과: {usage['allocated']:.2f} GB")
            return False
        
        return True
    
    def auto_adjust_batch_size(self) -> int:
        """VRAM 여유에 따라 배치 크기 자동 조정"""
        if not CUDA_AVAILABLE:
            return 1
        
        usage = self.get_vram_usage()
        allocated = usage["allocated"]
        
        if allocated < 6.0:
            return 4  # 여유 있음
        elif allocated < 8.0:
            return 3  # 정상
        else:
            return 2  # 부족
    
    def get_optimal_fps(self, video_duration_sec: float) -> int:
        """영상 길이에 따른 최적 FPS 결정"""
        if video_duration_sec > self.config.long_video_threshold_sec:
            return self.config.long_video_fps
        return self.config.target_fps
    
    # =========================================================================
    # 배치 처리
    # =========================================================================
    
    def process_batch(
        self,
        video_paths: List[str],
        processor: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        3개 영상 동시 처리
        
        Args:
            video_paths: 비디오 경로 목록
            processor: 처리 함수 (기본: MediaPipe 포즈 추출)
            
        Returns:
            처리 결과 목록
        """
        if processor is None:
            processor = self._default_processor
        
        start_time = time.time()
        batch_size = min(len(video_paths), self.auto_adjust_batch_size())
        results = []
        
        logger.info(f"🎬 배치 처리 시작: {len(video_paths)}개 영상 (배치 크기: {batch_size})")
        
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i+batch_size]
            
            # VRAM 확인
            if not self.check_vram_health():
                # 메모리 정리 후 재시도
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                batch_size = max(1, batch_size - 1)
            
            # 병렬 처리
            batch_results = self._process_batch_parallel(batch, processor)
            results.extend(batch_results)
            
            # 진행률 로깅
            progress = len(results) / len(video_paths) * 100
            logger.info(f"  진행률: {len(results)}/{len(video_paths)} ({progress:.1f}%)")
        
        elapsed = time.time() - start_time
        
        # 통계 업데이트
        with self._lock:
            self._stats["total_processed"] += len(results)
            self._stats["total_time_sec"] += elapsed
            
            if CUDA_AVAILABLE:
                peak_vram = torch.cuda.max_memory_allocated() / 1024**3
                self._stats["peak_vram_gb"] = max(self._stats["peak_vram_gb"], peak_vram)
        
        logger.info(f"✅ 배치 처리 완료: {len(results)}개, {elapsed:.1f}초")
        
        return results
    
    def _process_batch_parallel(
        self,
        batch: List[str],
        processor: Callable,
    ) -> List[Dict[str, Any]]:
        """배치 병렬 처리"""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            
            for stream_id, video_path in enumerate(batch):
                future = executor.submit(
                    self._process_single,
                    video_path,
                    stream_id,
                    processor,
                )
                futures.append(future)
            
            # 결과 수집
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"처리 실패: {e}")
        
        return results
    
    def _process_single(
        self,
        video_path: str,
        stream_id: int,
        processor: Callable,
    ) -> Optional[Dict[str, Any]]:
        """단일 스트림에서 처리"""
        try:
            if CUDA_AVAILABLE and stream_id < len(self._streams):
                with cuda.stream(self._streams[stream_id]):
                    return processor(video_path)
            else:
                return processor(video_path)
        except Exception as e:
            logger.error(f"스트림 {stream_id} 처리 실패: {e}")
            return None
    
    def _default_processor(self, video_path: str) -> Dict[str, Any]:
        """기본 프로세서 (MediaPipe 포즈 추출)"""
        try:
            from build_imitation_data import extract_pose_from_video, encode_imitation_data
            
            # 영상 길이 확인
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # 최적 FPS 결정
            target_fps = self.get_optimal_fps(duration)
            
            # 포즈 추출
            pose_data = extract_pose_from_video(video_path, output_fps=target_fps)
            
            if pose_data and pose_data.get("body") is not None:
                video_id = Path(video_path).stem
                il_data = encode_imitation_data(pose_data, video_id)
                
                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "success": True,
                    "frames": len(pose_data.get("body", [])),
                    "fps": target_fps,
                    "il_data": il_data,
                }
            
            return {
                "video_path": video_path,
                "success": False,
                "error": "포즈 추출 실패",
            }
            
        except Exception as e:
            return {
                "video_path": video_path,
                "success": False,
                "error": str(e),
            }
    
    # =========================================================================
    # 통계
    # =========================================================================
    
    @property
    def stats(self) -> Dict[str, Any]:
        """처리 통계"""
        return self._stats.copy()
    
    def print_stats(self):
        """통계 출력"""
        s = self._stats
        avg_time = s["total_time_sec"] / max(1, s["total_processed"])
        
        print(f"""
{'='*60}
📊 GPU 3-Stream 처리 통계
{'='*60}
  총 처리: {s['total_processed']}개
  총 시간: {s['total_time_sec']:.1f}초
  평균 처리 시간: {avg_time:.2f}초/영상
  피크 VRAM: {s['peak_vram_gb']:.2f} GB
""")


def monitor_gpu(interval_sec: float = 1.0, duration_sec: float = 60.0):
    """GPU 모니터링 (별도 스레드용)"""
    if not CUDA_AVAILABLE:
        print("CUDA 사용 불가")
        return
    
    start = time.time()
    
    while time.time() - start < duration_sec:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        # nvidia-smi로 사용률 조회
        try:
            import subprocess
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ])
            util = float(output.decode().strip())
        except:
            util = 0
        
        print(f"GPU: {util:.0f}% | VRAM: {allocated:.2f}/{reserved:.2f} GB")
        time.sleep(interval_sec)


if __name__ == "__main__":
    # 테스트
    manager = GPU3StreamManager()
    
    print("CUDA 사용 가능:", CUDA_AVAILABLE)
    print("VRAM:", manager.get_vram_usage())
    print("권장 배치 크기:", manager.auto_adjust_batch_size())
