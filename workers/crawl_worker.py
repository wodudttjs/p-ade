"""
멀티프로세스 크롤링 워커

Redis 큐에서 작업을 가져와 독립적으로 크롤링을 수행합니다.
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class WorkerConfig:
    """워커 설정"""
    worker_id: int
    timeout_sec: int = 5          # 큐 대기 시간
    max_tasks: int = 0            # 최대 처리 수 (0=무제한)
    source: str = "youtube"
    max_results_per_task: int = 50


def worker_loop(config: WorkerConfig):
    """
    워커 메인 루프
    
    Redis 큐에서 키워드를 가져와 크롤링을 수행합니다.
    """
    from task_queue.task_queue import CrawlTaskQueue
    from ingestion.multi_source_crawler import MultiSourceCrawler
    
    worker_id = config.worker_id
    logger.info(f"🚀 Worker {worker_id} 시작")
    
    queue = CrawlTaskQueue()
    if not queue.is_connected:
        logger.error(f"Worker {worker_id}: Redis 연결 실패")
        return
    
    # 크롤러 초기화 (캐시 사용)
    crawler = MultiSourceCrawler(
        sources=[config.source],
        max_results=config.max_results_per_task,
        max_workers=1,  # 단일 스레드
        use_cache=True,
    )
    
    completed = 0
    
    try:
        while True:
            # 최대 작업 수 체크
            if config.max_tasks > 0 and completed >= config.max_tasks:
                logger.info(f"Worker {worker_id}: 최대 작업 수 도달 ({completed})")
                break
            
            # 작업 가져오기
            queue.report_progress(worker_id, "waiting")
            task = queue.dequeue_keyword(timeout=config.timeout_sec)
            
            if not task:
                logger.info(f"Worker {worker_id}: 큐 비어있음, 종료")
                break
            
            keyword = task.get("keyword", "")
            source = task.get("source", config.source)
            
            logger.info(f"Worker {worker_id}: 크롤링 시작 - '{keyword}'")
            queue.report_progress(worker_id, "crawling", keyword)
            
            try:
                # 크롤링 실행
                results, stats = crawler.crawl(keywords=[keyword])
                
                # 결과 저장
                result_data = [
                    {
                        "video_id": r.video_id,
                        "url": r.url,
                        "title": r.title,
                        "duration_sec": r.duration_sec,
                    }
                    for r in results
                ]
                
                queue.mark_complete(keyword, result_data, source)
                logger.info(f"Worker {worker_id}: '{keyword}' 완료 - {len(results)}개")
                completed += 1
                
            except Exception as e:
                queue.mark_failed(keyword, str(e), source)
                logger.error(f"Worker {worker_id}: '{keyword}' 실패 - {e}")
    
    finally:
        queue.report_progress(worker_id, "stopped")
        logger.info(f"✅ Worker {worker_id} 종료 (처리: {completed}개)")


def run_workers(num_workers: int = None, keywords: List[str] = None, **kwargs):
    """
    멀티프로세스 크롤링 실행
    
    Args:
        num_workers: 워커 수 (기본: CPU 코어 수)
        keywords: 크롤링할 키워드 목록
        **kwargs: WorkerConfig 추가 옵션
    """
    from task_queue.task_queue import CrawlTaskQueue
    
    # CPU 코어 수 자동 감지
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    logger.info(f"🏭 멀티프로세스 크롤링 시작 (워커: {num_workers}개)")
    
    # 큐에 키워드 추가
    if keywords:
        queue = CrawlTaskQueue()
        if queue.is_connected:
            count = queue.enqueue_keywords(keywords, kwargs.get("source", "youtube"))
            logger.info(f"📋 {count}개 키워드 큐에 추가됨")
        else:
            logger.error("Redis 연결 실패")
            return
    
    # 워커 프로세스 시작
    processes = []
    for i in range(num_workers):
        config = WorkerConfig(worker_id=i, **kwargs)
        p = mp.Process(target=worker_loop, args=(config,))
        p.start()
        processes.append(p)
        logger.info(f"Worker {i} 프로세스 시작 (PID: {p.pid})")
    
    # 완료 대기
    for p in processes:
        p.join()
    
    # 결과 수집
    queue = CrawlTaskQueue()
    stats = queue.get_stats()
    
    logger.info(f"""
{'='*60}
📊 멀티프로세스 크롤링 완료
{'='*60}
  총 처리: {stats.get('total_completed', 0)}개
  총 결과: {stats.get('total_results', 0)}개
  실패: {stats.get('total_failed', 0)}개
""")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="멀티프로세스 크롤러")
    parser.add_argument("--workers", type=int, default=None, help="워커 수")
    parser.add_argument("--keywords", required=True, help="키워드 (콤마 구분)")
    parser.add_argument("--source", default="youtube", help="소스")
    parser.add_argument("--max-results", type=int, default=50, help="키워드당 최대 결과")
    
    args = parser.parse_args()
    
    keywords = [k.strip() for k in args.keywords.split(",")]
    
    run_workers(
        num_workers=args.workers,
        keywords=keywords,
        source=args.source,
        max_results_per_task=args.max_results,
    )
