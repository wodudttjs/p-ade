"""
배치 프로세싱 최적화

기능:
- GPU 병렬 처리
- 멀티프로세싱
- 작업 큐 관리
- 메모리 효율적 처리
"""

import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Optional, Dict
import queue
import time
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from extraction.pose_estimator import MediaPipePoseEstimator, VideoPoseSequence
from core.logging_config import logger


@dataclass
class BatchTask:
    """배치 작업"""
    task_id: str
    video_path: str
    output_path: str
    options: Dict = None


@dataclass
class BatchResult:
    """배치 결과"""
    task_id: str
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    num_frames: int = 0


class BatchPoseProcessor:
    """배치 포즈 프로세서"""
    
    def __init__(
        self,
        num_workers: int = None,
        gpu_ids: List[int] = None,
        model_complexity: int = 1,
    ):
        """
        Args:
            num_workers: 워커 프로세스 수 (None = CPU 코어 수)
            gpu_ids: 사용할 GPU ID 리스트 (None = CPU only)
            model_complexity: MediaPipe 모델 복잡도
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.gpu_ids = gpu_ids or []
        self.model_complexity = model_complexity
        
        # 작업 큐
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # 워커 프로세스
        self.workers: List[Process] = []
        self.is_running = False
    
    def start(self):
        """워커 시작"""
        self.is_running = True
        
        for i in range(self.num_workers):
            # GPU 할당 (순환)
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)] if self.gpu_ids else None
            
            worker = Process(
                target=self._worker_loop,
                args=(i, gpu_id),
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} workers")
    
    def stop(self):
        """워커 종료"""
        self.is_running = False
        
        # 종료 신호 전송
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # 워커 종료 대기
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()
        logger.info("All workers stopped")
    
    def _worker_loop(self, worker_id: int, gpu_id: Optional[int]):
        """워커 루프"""
        # GPU 설정
        if gpu_id is not None:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Worker {worker_id}: Using GPU {gpu_id}")
        else:
            logger.info(f"Worker {worker_id}: Using CPU")
        
        # MediaPipe 초기화 (각 워커마다 독립적)
        estimator = MediaPipePoseEstimator(
            model_complexity=self.model_complexity,
            enable_gpu=(gpu_id is not None),
        )
        
        while self.is_running:
            try:
                # 작업 가져오기 (1초 타임아웃)
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # 종료 신호
                    break
                
                logger.info(f"Worker {worker_id}: Processing {task.task_id}")
                
                # 작업 처리
                result = self._process_task(estimator, task)
                
                # 결과 전송
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error - {e}")
                result = BatchResult(
                    task_id=task.task_id if task else 'unknown',
                    success=False,
                    error_message=str(e),
                )
                self.result_queue.put(result)
    
    def _process_task(
        self,
        estimator: MediaPipePoseEstimator,
        task: BatchTask,
    ) -> BatchResult:
        """단일 작업 처리"""
        start_time = time.time()
        
        try:
            # 옵션 파싱
            options = task.options or {}
            output_fps = options.get('output_fps', 30)
            max_frames = options.get('max_frames', None)
            
            # 포즈 추정
            sequence = estimator.process_video(
                task.video_path,
                output_fps=output_fps,
                max_frames=max_frames,
            )
            
            # 저장
            estimator.save_sequence(sequence, task.output_path)
            
            processing_time = time.time() - start_time
            
            return BatchResult(
                task_id=task.task_id,
                success=True,
                output_path=task.output_path,
                processing_time=processing_time,
                num_frames=len(sequence.frames),
            )
        
        except Exception as e:
            return BatchResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )
    
    def submit_task(self, task: BatchTask):
        """작업 제출"""
        self.task_queue.put(task)
    
    def submit_batch(self, tasks: List[BatchTask]):
        """여러 작업 일괄 제출"""
        for task in tasks:
            self.submit_task(task)
        
        logger.info(f"Submitted {len(tasks)} tasks")
    
    def get_results(self, timeout: float = None) -> List[BatchResult]:
        """
        모든 결과 수집
        
        Args:
            timeout: 타임아웃 (초, None = 무한 대기)
        
        Returns:
            BatchResult 리스트
        """
        results = []
        start_time = time.time()
        
        while True:
            try:
                # 타임아웃 체크
                if timeout and (time.time() - start_time) > timeout:
                    break
                
                result = self.result_queue.get(timeout=1)
                results.append(result)
                
                logger.info(f"Received result: {result.task_id} - "
                          f"{'✓' if result.success else '✗'}")
                
            except queue.Empty:
                # 큐가 비어있으면 종료
                if not self.is_running or self.task_queue.empty():
                    break
        
        return results
    
    def wait_completion(self, expected_count: int, timeout: float = None) -> List[BatchResult]:
        """
        모든 작업 완료 대기
        
        Args:
            expected_count: 예상 결과 수
            timeout: 타임아웃
        """
        results = []
        start_time = time.time()
        
        logger.info(f"Waiting for {expected_count} results...")
        
        while len(results) < expected_count:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout: Only {len(results)}/{expected_count} completed")
                break
            
            try:
                result = self.result_queue.get(timeout=1)
                results.append(result)
                
                status = '✓' if result.success else '✗'
                logger.info(f"[{len(results)}/{expected_count}] {result.task_id} {status} "
                          f"({result.processing_time:.1f}s)")
                
            except queue.Empty:
                continue
        
        return results
    
    def get_statistics(self, results: List[BatchResult]) -> Dict:
        """결과 통계"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        processing_times = [r.processing_time for r in results if r.success]
        total_frames = sum(r.num_frames for r in results if r.success)
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_processing_time': sum(processing_times),
            'total_frames': total_frames,
            'avg_frames_per_video': total_frames / successful if successful > 0 else 0,
        }
    
    def print_statistics(self, results: List[BatchResult]):
        """통계 출력"""
        stats = self.get_statistics(results)
        
        logger.info("\n=== Batch Processing Statistics ===")
        logger.info(f"Total Tasks: {stats['total']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"Average Processing Time: {stats['avg_processing_time']:.1f}s")
        logger.info(f"Total Processing Time: {stats['total_processing_time']:.1f}s")
        logger.info(f"Total Frames Processed: {stats['total_frames']}")
        logger.info(f"Average Frames/Video: {stats['avg_frames_per_video']:.0f}")


class MemoryEfficientProcessor:
    """메모리 효율적 프로세서 (대용량 비디오용)"""
    
    def __init__(self, estimator: MediaPipePoseEstimator):
        self.estimator = estimator
    
    def process_video_streaming(
        self,
        video_path: str,
        output_path: str,
        chunk_size: int = 100,  # 한 번에 처리할 프레임 수
    ):
        """
        스트리밍 방식으로 비디오 처리 (메모리 절약)
        
        Args:
            video_path: 입력 비디오
            output_path: 출력 경로
            chunk_size: 청크 크기 (프레임)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 파일 준비 (append 모드)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_poses = []
        chunk_buffer = []
        frame_idx = 0
        
        logger.info(f"Processing {total_frames} frames in chunks of {chunk_size}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_idx / fps
            frame_pose = self.estimator.process_frame(frame, frame_idx, timestamp)
            
            if frame_pose:
                chunk_buffer.append(frame_pose)
            
            # 청크가 가득 차면 저장
            if len(chunk_buffer) >= chunk_size:
                all_poses.extend(chunk_buffer)
                chunk_buffer.clear()
                
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
            
            frame_idx += 1
        
        # 남은 버퍼 저장
        if chunk_buffer:
            all_poses.extend(chunk_buffer)
        
        cap.release()
        
        # 최종 저장
        sequence = VideoPoseSequence(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            frames=all_poses,
        )
        
        self.estimator.save_sequence(sequence, output_path)
        logger.info(f"Saved {len(all_poses)} frames to {output_path}")
