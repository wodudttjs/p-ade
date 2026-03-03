"""
AWS Lambda 서버리스 크롤러 함수

YouTube 및 다중 소스에서 로봇팔 영상을 크롤링하고 DynamoDB에 저장합니다.

Lambda 이벤트 형식:
{
    "keywords": ["robot arm", "pick place"],
    "max_per_keyword": 50,
    "sources": ["youtube"],
    "region": "us-east-1"
}

배포:
    1. pip install -t lambda_package/ -r lambda/requirements.txt
    2. cp lambda/crawler_function.py lambda_package/
    3. cd lambda_package && zip -r ../lambda_function.zip .
    4. aws lambda create-function --function-name robot-video-crawler ...
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# boto3는 Lambda 런타임에 기본 포함
import boto3
from botocore.exceptions import ClientError

# ============================================================
# 설정
# ============================================================

DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "robot-videos")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
MAX_RESULTS_DEFAULT = 50
SOURCES_DEFAULT = ["youtube"]

# DynamoDB 리소스 (Lambda 콜드 스타트 최적화를 위해 전역)
_dynamodb = None
_table = None


def _get_table():
    """DynamoDB 테이블 참조 (lazy init)"""
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        _table = _dynamodb.Table(DYNAMODB_TABLE)
    return _table


# ============================================================
# 크롤러 (Lambda 내장용 경량 버전)
# ============================================================

class LambdaCrawler:
    """
    Lambda 환경용 경량 크롤러

    yt-dlp가 Lambda 패키지에 포함된 경우 사용하고,
    없으면 YouTube Data API v3 또는 웹 스크래핑 폴백.
    """

    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self._yt_dlp_available = False
        try:
            import yt_dlp  # noqa: F401
            self._yt_dlp_available = True
        except ImportError:
            pass

    def search(self, keyword: str, source: str = "youtube") -> List[Dict[str, Any]]:
        """키워드로 영상 검색"""
        if self._yt_dlp_available:
            return self._search_yt_dlp(keyword, source)
        return self._search_api(keyword)

    def _search_yt_dlp(self, keyword: str, source: str) -> List[Dict[str, Any]]:
        """yt-dlp 기반 검색"""
        import yt_dlp

        search_url = f"ytsearch{self.max_results}:{keyword}"
        if source == "google_videos":
            search_url = f"gvsearch{self.max_results}:{keyword}"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "ignoreerrors": True,
        }

        results = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_url, download=False)
                entries = info.get("entries", []) if info else []

                for entry in entries:
                    if not entry or not entry.get("id"):
                        continue
                    results.append({
                        "video_id": entry.get("id", ""),
                        "title": entry.get("title", ""),
                        "url": entry.get("url") or entry.get("webpage_url")
                              or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                        "duration": entry.get("duration"),
                        "view_count": entry.get("view_count"),
                        "channel": entry.get("channel") or entry.get("uploader", ""),
                        "platform": source,
                    })
        except Exception:
            pass

        return results

    def _search_api(self, keyword: str) -> List[Dict[str, Any]]:
        """YouTube Data API v3 폴백 (API 키 필요)"""
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return []

        try:
            import requests  # noqa: F811
        except ImportError:
            from urllib import request as urllib_request
            import json as json_mod

            url = (
                f"https://www.googleapis.com/youtube/v3/search"
                f"?part=snippet&q={keyword}&type=video"
                f"&maxResults={min(self.max_results, 50)}"
                f"&key={api_key}"
            )
            try:
                resp = urllib_request.urlopen(url, timeout=30)
                data = json_mod.loads(resp.read().decode())
                results = []
                for item in data.get("items", []):
                    vid = item["id"].get("videoId", "")
                    snippet = item.get("snippet", {})
                    results.append({
                        "video_id": vid,
                        "title": snippet.get("title", ""),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "channel": snippet.get("channelTitle", ""),
                        "platform": "youtube",
                    })
                return results
            except Exception:
                return []

        # requests 사용 가능
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": min(self.max_results, 50),
            "key": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("items", []):
                vid = item["id"].get("videoId", "")
                snippet = item.get("snippet", {})
                results.append({
                    "video_id": vid,
                    "title": snippet.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "channel": snippet.get("channelTitle", ""),
                    "platform": "youtube",
                })
            return results
        except Exception:
            return []


# ============================================================
# DynamoDB 저장
# ============================================================

def save_to_dynamodb(
    videos: List[Dict[str, Any]],
    keyword: str,
    table=None,
) -> int:
    """
    크롤링 결과를 DynamoDB에 배치 저장

    Args:
        videos: 영상 메타데이터 리스트
        keyword: 검색 키워드
        table: DynamoDB Table 객체 (테스트용 주입)

    Returns:
        저장된 아이템 수
    """
    if table is None:
        table = _get_table()

    saved = 0
    now = datetime.now(timezone.utc).isoformat()

    with table.batch_writer() as batch:
        for video in videos:
            vid = video.get("video_id", "")
            if not vid:
                continue
            try:
                item = {
                    "video_id": vid,
                    "keyword": keyword,
                    "title": video.get("title", ""),
                    "url": video.get("url", ""),
                    "metadata": json.dumps(video, ensure_ascii=False, default=str),
                    "collected": False,
                    "platform": video.get("platform", "youtube"),
                    "created_at": now,
                }
                # DynamoDB는 빈 문자열 허용 안 함 → None 변환
                item = {k: v for k, v in item.items() if v is not None and v != ""}
                item.setdefault("video_id", vid)
                item.setdefault("collected", False)

                batch.put_item(Item=item)
                saved += 1
            except Exception:
                continue

    return saved


# ============================================================
# Lambda 핸들러
# ============================================================

def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AWS Lambda 진입 함수

    Args:
        event: {
            "keywords": ["robot arm", "pick place"],
            "max_per_keyword": 50,
            "sources": ["youtube"],
        }
        context: Lambda 컨텍스트 (자동 주입)

    Returns:
        {
            "statusCode": 200,
            "body": {
                "keywords_processed": 2,
                "videos_found": 85,
                "errors": [],
                "duration_sec": 12.3
            }
        }
    """
    start_time = time.time()

    # 입력 파싱
    keywords = event.get("keywords", [])
    max_results = event.get("max_per_keyword", MAX_RESULTS_DEFAULT)
    sources = event.get("sources", SOURCES_DEFAULT)

    if not keywords:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No keywords provided"}),
        }

    crawler = LambdaCrawler(max_results=max_results)
    total_found = 0
    errors = []
    keyword_results = {}

    for keyword in keywords:
        kw_total = 0
        for source in sources:
            try:
                videos = crawler.search(keyword, source)
                if videos:
                    saved = save_to_dynamodb(videos, keyword)
                    kw_total += saved
            except Exception as e:
                errors.append({
                    "keyword": keyword,
                    "source": source,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        keyword_results[keyword] = kw_total
        total_found += kw_total

    elapsed = round(time.time() - start_time, 2)

    result = {
        "keywords_processed": len(keywords),
        "videos_found": total_found,
        "keyword_results": keyword_results,
        "errors": errors,
        "duration_sec": elapsed,
    }

    return {
        "statusCode": 200,
        "body": json.dumps(result, ensure_ascii=False),
    }


# ============================================================
# DynamoDB 테이블 생성 헬퍼
# ============================================================

def create_dynamodb_table(
    table_name: str = DYNAMODB_TABLE,
    region: str = AWS_REGION,
) -> Dict[str, Any]:
    """
    DynamoDB 테이블 생성

    Args:
        table_name: 테이블 이름
        region: AWS 리전

    Returns:
        테이블 정보 dict
    """
    dynamodb = boto3.resource("dynamodb", region_name=region)

    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "video_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "video_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
        return {"status": "created", "table_name": table_name}
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            return {"status": "already_exists", "table_name": table_name}
        raise


# ============================================================
# CLI (로컬 테스트용)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lambda 크롤러 로컬 테스트")
    parser.add_argument("--keywords", nargs="+", default=["robot arm"])
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="DynamoDB 저장 안 함")
    args = parser.parse_args()

    if args.dry_run:
        # DynamoDB 없이 크롤링만 테스트
        crawler = LambdaCrawler(max_results=args.max_results)
        for kw in args.keywords:
            results = crawler.search(kw)
            print(f"[{kw}] → {len(results)}개 발견")
            for v in results[:3]:
                print(f"  - {v['title'][:60]}")
    else:
        event = {
            "keywords": args.keywords,
            "max_per_keyword": args.max_results,
        }
        response = lambda_handler(event)
        print(json.dumps(json.loads(response["body"]), indent=2, ensure_ascii=False))
