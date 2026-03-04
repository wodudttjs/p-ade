"""
CPUWorkerPool — 긴 영상(60초 이상) 전용 CPU 멀티프로세스 처리기

목적:
  - 60초 이상 긴 영상이 GPU 스트림을 점유하는 병목 방지
  - CPU ProcessPoolExecutor로 별도 처리
  - MediaPipe CPU 모드 + 15fps 다운샘플링

StreamManager.process_batch()와 연동:
  - 60초 미만 → GPU 6-Stream (GPU3StreamManager)
  - 60초 이상 → CPUWorkerPool
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)

# CPU 워커 수: 물리 코어의 절반
_CPU_COUNT = max(1, (os.cpu_count() or 4) // 2)

# 긴 영상 다운샘플 FPS
_LONG_VIDEO_FPS = 15.0


def _process_long_video(args: tuple) -> Dict[str, Any]:
    """
    긴 영상 처리 함수 (별도 프로세스에서 실행)

    MediaPipe CPU 모드, 15fps 다운샘플링으로 포즈 추출 + IL 인코딩.
    ProcessPoolExecutor에서 호출되므로 최상위 함수여야 합니다.
    """
    video_path, output_dir = args
    start_time = time.time()
    video_id = Path(video_path).stem

    try:
        # output 경로
        out_path = None
        if output_dir:
            out_path = Path(output_dir) / f"{video_id}_episode.npz"
            if out_path.exists():
                return {
                    "video_id": video_id,
                    "video_path": video_path,
                    "success": True,
                    "status": "skipped",
                    "frames": 0,
                    "elapsed_sec": 0.0,
                    "processor": "cpu",
                }

        # UnifiedVideoProcessor CPU 모드로 처리
        sys.path.insert(0, str(PROJECT_ROOT))
        from gpu.unified_processor import UnifiedVideoProcessor

        processor = UnifiedVideoProcessor(
            output_fps=_LONG_VIDEO_FPS,
            device="cpu",   # CPU 모드 강제
        )
        result = processor.process(video_path, str(out_path) if out_path else None)
        result["processor"] = "cpu"
        result["elapsed_sec"] = time.time() - start_time
        return result

    except Exception as e:
        return {
            "video_id": video_id,
            "video_path": video_path,
            "success": False,
            "status": "failed",
            "error": str(e),
            "elapsed_sec": time.time() - start_time,
            "processor": "cpu",
        }


class CPUWorkerPool:
    """
    긴 영상 전용 CPU 멀티프로세스 풀

    사용법:
        pool = CPUWorkerPool(max_workers=4)
        results = pool.process(long_video_paths, output_dir="data/episodes")
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Args:
            max_workers: CPU 워커 수 (기본: CPU_COUNT // 2)
        """
        self.max_workers = max_workers or _CPU_COUNT
        logger.info(f"CPUWorkerPool 초기화: max_workers={self.max_workers}, fps={_LONG_VIDEO_FPS}")

    def process(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        긴 영상들을 CPU 멀티프로세스로 처리

        Args:
            video_paths: 비디오 경로 목록 (60초 이상 영상)
            output_dir: NPZ 저장 디렉토리

        Returns:
            처리 결과 목록 (입력 순서 유지)
        """
        if not video_paths:
            return []

        total = len(video_paths)
        logger.info(f"🖥️ CPU WorkerPool: {total}개 긴 영상 처리 시작 (workers={self.max_workers})")

        start_time = time.time()
        results_map: Dict[str, Dict] = {}

        # 작업 목록 생성
        tasks = [(vp, output_dir) for vp in video_paths]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(_process_long_video, task): task[0]
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_path):
                video_path = future_to_path[future]
                completed += 1
                try:
                    result = future.result(timeout=600)  # 10분 타임아웃
                    results_map[video_path] = result
                    status = result.get("status", "done")
                    elapsed = result.get("elapsed_sec", 0)
                    logger.info(
                        f"  🖥️ [{completed}/{total}] {Path(video_path).stem}: "
                        f"{status} ({elapsed:.1f}s)"
                    )
                except Exception as e:
                    results_map[video_path] = {
                        "video_id": Path(video_path).stem,
                        "video_path": video_path,
                        "success": False,
                        "status": "failed",
                        "error": str(e),
                        "processor": "cpu",
                    }

        # 입력 순서대로 결과 반환
        results = [results_map.get(vp, {
            "video_id": Path(vp).stem,
            "video_path": vp,
            "success": False,
            "status": "failed",
            "error": "결과 없음",
        }) for vp in video_paths]

        total_elapsed = time.time() - start_time
        success = sum(1 for r in results if r.get("success"))
        logger.info(
            f"🖥️ CPU WorkerPool 완료: {success}/{total} 성공, "
            f"총 {total_elapsed:.1f}초 ({total_elapsed/max(total,1):.1f}초/영상)"
        )

        return results

    def get_video_duration(self, video_path: str) -> float:
        """비디오 길이(초) 반환"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frames / fps if fps > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def split_by_duration(
        video_paths: List[str],
        threshold_sec: float = 60.0,
    ):
        """
        영상 길이로 GPU/CPU 분리

        Returns:
            (short_paths, long_paths)
        """
        short, long = [], []
        for vp in video_paths:
            try:
                import cv2
                cap = cv2.VideoCapture(vp)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frames / fps if fps > 0 else 0.0
                cap.release()
                if duration > threshold_sec:
                    long.append(vp)
                else:
                    short.append(vp)
            except Exception:
                short.append(vp)  # 확인 실패 시 GPU로 처리
        return short, long
