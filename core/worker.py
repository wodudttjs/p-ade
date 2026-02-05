"""
Worker Process - 병렬 작업 처리 워커

MVP Phase 2 Week 5: Queue System
- 병렬 다운로드 (4 workers)
- 처리 속도 2배 개선
"""

import os
import sys
import time
import signal
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logger
from core.queue_manager import (
    QueueManager,
    Task,
    TaskStatus,
    TaskPriority,
    InMemoryQueue,
)

logger = setup_logger(__name__)


class WorkerType(Enum):
    """워커 타입"""
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    UPLOAD = "upload"


@dataclass
class WorkerConfig:
    """워커 설정"""
    worker_type: WorkerType
    num_workers: int = 4
    poll_interval: float = 1.0  # 초
    max_tasks_per_worker: int = 100
    timeout_per_task: int = 300  # 5분


class TaskProcessor:
    """
    작업 처리기
    
    각 작업 타입별 실제 처리 로직을 담당
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
    
    def process_download(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """다운로드 작업 처리"""
        video_id = payload.get("video_id")
        url = payload.get("url")
        
        logger.info(f"⬇️ 다운로드 시작: {video_id}")
        
        try:
            # 실제 다운로드 로직 (yt-dlp)
            import subprocess
            
            output_dir = self.base_dir / "raw"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_id}.mp4"
            
            # 이미 존재하면 스킵
            if output_path.exists():
                logger.info(f"⏭️ 이미 존재: {output_path}")
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "skipped": True,
                }
            
            # yt-dlp로 다운로드
            cmd = [
                "yt-dlp",
                "-f", "best[height<=720]",
                "-o", str(output_path),
                url,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=payload.get("timeout", 300),
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"✅ 다운로드 완료: {output_path}")
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "size_bytes": output_path.stat().st_size,
                }
            else:
                raise Exception(f"yt-dlp 실패: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("다운로드 타임아웃")
        except Exception as e:
            logger.error(f"❌ 다운로드 실패: {e}")
            raise
    
    def process_extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 추출 작업 처리"""
        video_id = payload.get("video_id")
        video_path = payload.get("video_path")
        
        logger.info(f"🦴 포즈 추출 시작: {video_id}")
        
        try:
            import cv2
            import numpy as np
            import mediapipe as mp
            
            output_dir = self.base_dir / "poses"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_id}_pose.npz"
            
            # 이미 존재하면 스킵
            if output_path.exists():
                logger.info(f"⏭️ 이미 존재: {output_path}")
                return {
                    "success": True,
                    "pose_path": str(output_path),
                    "skipped": True,
                }
            
            # MediaPipe Pose 초기화
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            
            # 비디오 읽기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"비디오 열기 실패: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            poses = []
            confidences = []
            frame_indices = []
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # BGR -> RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # 랜드마크 추출
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    poses.append(landmarks)
                    confidences.append(
                        np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
                    )
                    frame_indices.append(frame_idx)
                
                frame_idx += 1
            
            cap.release()
            pose.close()
            
            if not poses:
                raise Exception("포즈 감지 실패")
            
            # 저장
            np.savez_compressed(
                output_path,
                poses=np.array(poses, dtype=np.float32),
                confidences=np.array(confidences, dtype=np.float32),
                frame_indices=np.array(frame_indices, dtype=np.int32),
                fps=fps,
                total_frames=total_frames,
                video_id=video_id,
            )
            
            logger.info(f"✅ 포즈 추출 완료: {len(poses)} 프레임")
            
            return {
                "success": True,
                "pose_path": str(output_path),
                "num_frames": len(poses),
                "avg_confidence": float(np.mean(confidences)),
            }
            
        except Exception as e:
            logger.error(f"❌ 포즈 추출 실패: {e}")
            raise
    
    def process_transform(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """변환 작업 처리"""
        video_id = payload.get("video_id")
        pose_path = payload.get("pose_path")
        
        logger.info(f"🔄 변환 시작: {video_id}")
        
        try:
            import numpy as np
            
            output_dir = self.base_dir / "episodes"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 포즈 데이터 로드
            data = np.load(pose_path)
            poses = data["poses"]
            confidences = data["confidences"]
            fps = float(data["fps"])
            
            # 에피소드 분할 (간단한 구현)
            # 실제로는 segment_episodes.py의 로직 사용
            min_episode_frames = int(fps * 2)  # 최소 2초
            
            episodes = []
            current_start = 0
            
            for i, conf in enumerate(confidences):
                if conf < 0.3 and i - current_start >= min_episode_frames:
                    episodes.append({
                        "start_frame": current_start,
                        "end_frame": i,
                        "avg_confidence": float(np.mean(confidences[current_start:i])),
                    })
                    current_start = i + 1
            
            # 마지막 에피소드
            if len(poses) - current_start >= min_episode_frames:
                episodes.append({
                    "start_frame": current_start,
                    "end_frame": len(poses),
                    "avg_confidence": float(np.mean(confidences[current_start:])),
                })
            
            logger.info(f"✅ 변환 완료: {len(episodes)} 에피소드")
            
            return {
                "success": True,
                "num_episodes": len(episodes),
                "episodes": episodes,
            }
            
        except Exception as e:
            logger.error(f"❌ 변환 실패: {e}")
            raise
    
    def process_upload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """업로드 작업 처리"""
        video_id = payload.get("video_id")
        file_path = payload.get("file_path")
        
        logger.info(f"☁️ 업로드 시작: {video_id}")
        
        try:
            from storage.providers.s3_provider import S3Provider
            from config.settings import Config
            
            config = Config()
            s3 = S3Provider(
                bucket_name=config.get("AWS_S3_BUCKET", "p-ade-datasets"),
                region=config.get("AWS_REGION", "us-east-1"),
            )
            
            # S3 키 생성
            now = datetime.now()
            filename = Path(file_path).name
            s3_key = f"poses/{now.year}/{now.month:02d}/{now.day:02d}/{filename}"
            
            # 업로드
            s3.upload_file(file_path, s3_key)
            
            s3_uri = f"s3://{s3.bucket_name}/{s3_key}"
            logger.info(f"✅ 업로드 완료: {s3_uri}")
            
            return {
                "success": True,
                "s3_uri": s3_uri,
                "s3_key": s3_key,
            }
            
        except Exception as e:
            logger.error(f"❌ 업로드 실패: {e}")
            raise
    
    def process(self, task: Task) -> Dict[str, Any]:
        """작업 타입에 따라 처리"""
        handlers = {
            "download": self.process_download,
            "extract": self.process_extract,
            "transform": self.process_transform,
            "upload": self.process_upload,
        }
        
        handler = handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"알 수 없는 작업 타입: {task.task_type}")
        
        return handler(task.payload)


class Worker:
    """
    워커 프로세스
    
    큐에서 작업을 가져와 처리
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        queue_name: str,
        processor: TaskProcessor,
    ):
        self.worker_id = worker_id
        self.qm = queue_manager
        self.queue_name = queue_name
        self.processor = processor
        self._running = False
    
    def start(self):
        """워커 시작"""
        self._running = True
        logger.info(f"🟢 Worker {self.worker_id} 시작 (queue={self.queue_name})")
        
        while self._running:
            try:
                # 작업 가져오기
                task = self.qm.get_next_task(self.queue_name, timeout=1)
                
                if task:
                    task.worker_id = self.worker_id
                    logger.info(f"📥 Worker {self.worker_id}: 작업 수신 - {task.task_id}")
                    
                    try:
                        # 작업 처리
                        result = self.processor.process(task)
                        self.qm.complete_task(self.queue_name, task, result)
                        logger.info(f"✅ Worker {self.worker_id}: 작업 완료 - {task.task_id}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        self.qm.fail_task(self.queue_name, task, error_msg, retry=True)
                        logger.error(f"❌ Worker {self.worker_id}: 작업 실패 - {error_msg}")
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} 오류: {e}")
                time.sleep(1)
        
        logger.info(f"🔴 Worker {self.worker_id} 종료")
    
    def stop(self):
        """워커 중지"""
        self._running = False


class WorkerPool:
    """
    워커 풀
    
    여러 워커를 병렬로 실행
    """
    
    def __init__(
        self,
        queue_manager: QueueManager,
        num_workers: int = 4,
        base_dir: str = "data",
    ):
        self.qm = queue_manager
        self.num_workers = num_workers
        self.processor = TaskProcessor(base_dir)
        self.workers: List[Worker] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
    
    def start(self, queue_name: str):
        """워커 풀 시작"""
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info(f"🚀 WorkerPool 시작: {self.num_workers} workers (queue={queue_name})")
        
        futures = []
        for i in range(self.num_workers):
            worker = Worker(
                worker_id=f"worker-{i+1}",
                queue_manager=self.qm,
                queue_name=queue_name,
                processor=self.processor,
            )
            self.workers.append(worker)
            futures.append(self._executor.submit(worker.start))
        
        return futures
    
    def stop(self):
        """워커 풀 중지"""
        logger.info("🛑 WorkerPool 중지 중...")
        self._running = False
        
        for worker in self.workers:
            worker.stop()
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        self.workers.clear()
        logger.info("✅ WorkerPool 중지 완료")
    
    def get_stats(self) -> Dict[str, Any]:
        """풀 통계"""
        return {
            "num_workers": self.num_workers,
            "active_workers": len([w for w in self.workers if w._running]),
            "queue_stats": self.qm.get_queue_stats(),
        }


def process_batch_parallel(
    items: List[Dict[str, Any]],
    task_type: str,
    num_workers: int = 4,
    use_redis: bool = False,
) -> List[Dict[str, Any]]:
    """
    배치 작업을 병렬로 처리
    
    Args:
        items: 처리할 항목 목록
        task_type: 작업 타입 (download, extract, transform, upload)
        num_workers: 워커 수
        use_redis: Redis 사용 여부
    
    Returns:
        결과 목록
    """
    qm = QueueManager(use_redis=use_redis)
    processor = TaskProcessor()
    
    # 작업 제출
    logger.info(f"📤 {len(items)}개 작업 제출 중...")
    
    task_ids = []
    for item in items:
        task = qm.create_task(task_type, item)
        qm.backend.enqueue(task_type, task)
        task_ids.append(task.task_id)
    
    # 병렬 처리
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        def process_one():
            task = qm.get_next_task(task_type)
            if task:
                try:
                    result = processor.process(task)
                    qm.complete_task(task_type, task, result)
                    return {"task_id": task.task_id, "success": True, **result}
                except Exception as e:
                    qm.fail_task(task_type, task, str(e), retry=False)
                    return {"task_id": task.task_id, "success": False, "error": str(e)}
            return None
        
        futures = [executor.submit(process_one) for _ in items]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    success_count = len([r for r in results if r.get("success")])
    logger.info(f"✅ 완료: {success_count}/{len(items)} 성공")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="P-ADE Worker")
    parser.add_argument("--queue", default="download", choices=["download", "extract", "transform", "upload"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--redis", action="store_true", help="Redis 사용")
    args = parser.parse_args()
    
    print(f"=== P-ADE Worker Pool ===")
    print(f"Queue: {args.queue}")
    print(f"Workers: {args.workers}")
    print(f"Redis: {args.redis}")
    print()
    
    # 큐 매니저 생성
    qm = QueueManager(use_redis=args.redis)
    
    # 워커 풀 시작
    pool = WorkerPool(qm, num_workers=args.workers)
    
    # 시그널 핸들러
    def signal_handler(sig, frame):
        print("\n⚠️ 종료 신호 수신...")
        pool.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        futures = pool.start(args.queue)
        
        # 완료 대기
        for future in as_completed(futures):
            pass
            
    except KeyboardInterrupt:
        pool.stop()
