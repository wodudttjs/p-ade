"""
Lambda 배치 호출 스크립트

로컬에서 AWS Lambda 크롤러를 대규모 병렬 호출합니다.

사용법:
    python lambda/invoke_lambda.py --keywords-file data/urls_mass.csv --batch-size 10
    python lambda/invoke_lambda.py --generate --batch-size 10 --max-batches 5
    python lambda/invoke_lambda.py --keywords "robot arm" "pick place" "cobot"
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================
# 설정
# ============================================================

@dataclass
class LambdaInvokeConfig:
    """Lambda 호출 설정"""
    function_name: str = "robot-video-crawler"
    region: str = os.environ.get("AWS_REGION", "ap-northeast-2")
    batch_size: int = 10
    max_per_keyword: int = 50
    max_concurrent: int = 100  # Lambda 동시 실행 제한
    invocation_type: str = "Event"  # Event=비동기, RequestResponse=동기
    sources: List[str] = field(default_factory=lambda: ["youtube"])
    delay_between_batches: float = 0.5  # 초


# ============================================================
# Lambda 호출기
# ============================================================

class LambdaInvoker:
    """AWS Lambda 함수 배치 호출기"""

    def __init__(self, config: Optional[LambdaInvokeConfig] = None):
        self.config = config or LambdaInvokeConfig()
        self._client = None
        self._results: List[Dict[str, Any]] = []

    @property
    def client(self):
        """Lambda 클라이언트 (lazy init)"""
        if self._client is None:
            self._client = boto3.client(
                "lambda",
                region_name=self.config.region,
            )
        return self._client

    def invoke(self, keywords: List[str]) -> Dict[str, Any]:
        """
        단일 Lambda 호출

        Args:
            keywords: 키워드 리스트

        Returns:
            Lambda 응답 dict
        """
        payload = {
            "keywords": keywords,
            "max_per_keyword": self.config.max_per_keyword,
            "sources": self.config.sources,
        }

        try:
            response = self.client.invoke(
                FunctionName=self.config.function_name,
                InvocationType=self.config.invocation_type,
                Payload=json.dumps(payload),
            )

            result = {
                "status_code": response.get("StatusCode", 0),
                "keywords": keywords,
                "invocation_type": self.config.invocation_type,
                "timestamp": datetime.now().isoformat(),
            }

            # 동기 호출인 경우 응답 본문 읽기
            if self.config.invocation_type == "RequestResponse":
                payload_stream = response.get("Payload")
                if payload_stream:
                    body = json.loads(payload_stream.read().decode())
                    result["response"] = body

            self._results.append(result)
            return result

        except ClientError as e:
            error_result = {
                "status_code": 0,
                "keywords": keywords,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._results.append(error_result)
            return error_result

    def parallel_invoke(
        self,
        all_keywords: List[str],
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        키워드를 배치 분할하여 Lambda 병렬 호출

        Args:
            all_keywords: 전체 키워드 리스트
            batch_size: 배치당 키워드 수
            max_batches: 최대 배치 수 (None=전부)

        Returns:
            호출 결과 요약
        """
        bs = batch_size or self.config.batch_size

        # 배치 분할
        batches = [
            all_keywords[i:i + bs]
            for i in range(0, len(all_keywords), bs)
        ]

        if max_batches:
            batches = batches[:max_batches]

        total_batches = len(batches)
        total_keywords = sum(len(b) for b in batches)

        logger.info(
            f"🚀 Lambda 병렬 호출 시작: {total_batches}개 배치, "
            f"{total_keywords}개 키워드"
        )

        start_time = time.time()
        success_count = 0
        error_count = 0

        for i, batch in enumerate(batches):
            result = self.invoke(batch)

            if result.get("status_code") in (200, 202):
                success_count += 1
            else:
                error_count += 1

            logger.info(
                f"  배치 {i + 1}/{total_batches} 호출 완료 "
                f"({len(batch)}개 키워드)"
            )

            # 동시 실행 제한을 위한 딜레이
            if self.config.delay_between_batches > 0:
                time.sleep(self.config.delay_between_batches)

        elapsed = round(time.time() - start_time, 2)

        summary = {
            "total_batches": total_batches,
            "total_keywords": total_keywords,
            "success": success_count,
            "errors": error_count,
            "elapsed_sec": elapsed,
            "invocation_type": self.config.invocation_type,
        }

        logger.info(
            f"✅ Lambda 호출 완료: {success_count}/{total_batches} 성공, "
            f"{elapsed}초 소요"
        )

        return summary

    def get_results(self) -> List[Dict[str, Any]]:
        """호출 결과 목록 반환"""
        return list(self._results)

    def check_function_exists(self) -> bool:
        """Lambda 함수 존재 여부 확인"""
        try:
            self.client.get_function(
                FunctionName=self.config.function_name
            )
            return True
        except ClientError:
            return False


# ============================================================
# 키워드 로더
# ============================================================

def load_keywords_from_file(filepath: str) -> List[str]:
    """파일에서 키워드 로드 (CSV 또는 텍스트)"""
    path = Path(filepath)
    keywords = []

    if path.suffix == ".csv":
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    keywords.append(row[0].strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keywords.append(line)

    return keywords


def load_keywords_from_generator(max_count: int = 500) -> List[str]:
    """KeywordGenerator에서 키워드 생성"""
    try:
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en", "ko"])
        return gen.get_flat_keywords(max_count=max_count)
    except ImportError:
        logger.warning("KeywordGenerator를 불러올 수 없습니다.")
        return []


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lambda 크롤러 배치 호출")
    parser.add_argument("--keywords", nargs="+", help="직접 키워드 지정")
    parser.add_argument("--keywords-file", help="키워드 파일 경로 (CSV/TXT)")
    parser.add_argument("--generate", action="store_true",
                        help="KeywordGenerator로 자동 생성")
    parser.add_argument("--max-keywords", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-per-keyword", type=int, default=50)
    parser.add_argument("--function-name", default="robot-video-crawler")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-northeast-2"))
    parser.add_argument("--sync", action="store_true",
                        help="호출 후 비동기 → 동기 모드로 전환")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 호출 없이 배치 계획만 출력")
    args = parser.parse_args()

    # 키워드 수집
    keywords = []
    if args.keywords:
        keywords = args.keywords
    elif args.keywords_file:
        keywords = load_keywords_from_file(args.keywords_file)
    elif args.generate:
        keywords = load_keywords_from_generator(args.max_keywords)
    else:
        parser.error("--keywords, --keywords-file, 또는 --generate 중 하나 필요")

    logger.info(f"총 {len(keywords)}개 키워드 로드됨")

    if args.dry_run:
        batches = [
            keywords[i:i + args.batch_size]
            for i in range(0, len(keywords), args.batch_size)
        ]
        if args.max_batches:
            batches = batches[:args.max_batches]
        print(f"\n📋 드라이런 — {len(batches)}개 배치, {sum(len(b) for b in batches)}개 키워드")
        for i, batch in enumerate(batches[:5]):
            print(f"  배치 {i + 1}: {batch[:3]}{'...' if len(batch) > 3 else ''}")
        if len(batches) > 5:
            print(f"  ... 외 {len(batches) - 5}개 배치")
        return

    # Lambda 호출
    config = LambdaInvokeConfig(
        function_name=args.function_name,
        region=args.region,
        batch_size=args.batch_size,
        max_per_keyword=args.max_per_keyword,
        invocation_type="RequestResponse" if args.sync else "Event",
    )

    invoker = LambdaInvoker(config)

    if not invoker.check_function_exists():
        logger.error(f"Lambda 함수 '{args.function_name}'을 찾을 수 없습니다.")
        sys.exit(1)

    summary = invoker.parallel_invoke(
        keywords,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
