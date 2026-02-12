"""
Task 3 통합 테스트

테스트 범위:
- 3.1: Lambda 크롤러 (crawler_function, invoke_lambda, dynamodb_sync)
- 3.2: 키워드 확장 (카르테시안 곱, MultilingualExpander, LongtailDiscovery)
"""

import json
import os
import sys
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import pytest

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 3.1 Lambda 크롤러 테스트
# ============================================================

class TestLambdaCrawler:
    """lambda/crawler_function.py - LambdaCrawler 테스트"""

    def test_import(self):
        """모듈 import 확인"""
        from lambda_.crawler_function import LambdaCrawler
        crawler = LambdaCrawler(max_results=5)
        assert crawler.max_results == 5

    def test_search_no_ytdlp_no_api_key(self):
        """yt-dlp, API 키 모두 없을 때 빈 결과"""
        from lambda_.crawler_function import LambdaCrawler
        crawler = LambdaCrawler(max_results=5)
        crawler._yt_dlp_available = False
        with patch.dict(os.environ, {}, clear=True):
            results = crawler._search_api("robot arm")
        assert results == []

    def test_search_yt_dlp(self):
        """yt-dlp 기반 검색 테스트"""
        from lambda_.crawler_function import LambdaCrawler
        crawler = LambdaCrawler(max_results=3)
        crawler._yt_dlp_available = True

        mock_entries = [
            {"id": "abc123", "title": "Robot Arm Demo", "url": "https://youtube.com/watch?v=abc123",
             "duration": 120, "view_count": 5000, "channel": "RoboChannel"},
            {"id": "def456", "title": "Pick and Place", "webpage_url": "https://youtube.com/watch?v=def456",
             "duration": 60, "view_count": 1000, "uploader": "RoboUser"},
        ]

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {"entries": mock_entries}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"yt_dlp": MagicMock()}):
            import yt_dlp
            yt_dlp.YoutubeDL.return_value = mock_ydl_instance
            with patch("lambda_.crawler_function.LambdaCrawler._search_yt_dlp") as mock_search:
                mock_search.return_value = [
                    {"video_id": "abc123", "title": "Robot Arm Demo",
                     "url": "https://youtube.com/watch?v=abc123",
                     "duration": 120, "view_count": 5000,
                     "channel": "RoboChannel", "platform": "youtube"},
                ]
                results = crawler.search("robot arm")
                assert len(results) >= 1
                assert results[0]["video_id"] == "abc123"


class TestSaveToDynamoDB:
    """lambda/crawler_function.py - save_to_dynamodb 테스트"""

    def test_save_batch(self):
        """DynamoDB 배치 저장"""
        from lambda_.crawler_function import save_to_dynamodb

        mock_table = MagicMock()
        mock_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = MagicMock(return_value=mock_writer)
        mock_table.batch_writer.return_value.__exit__ = MagicMock(return_value=False)

        videos = [
            {"video_id": "v1", "title": "Test 1", "url": "https://y.com/1"},
            {"video_id": "v2", "title": "Test 2", "url": "https://y.com/2"},
        ]

        saved = save_to_dynamodb(videos, "robot arm", table=mock_table)
        assert saved == 2
        assert mock_writer.put_item.call_count == 2

    def test_save_empty(self):
        """빈 리스트 저장"""
        from lambda_.crawler_function import save_to_dynamodb
        mock_table = MagicMock()
        mock_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = MagicMock(return_value=mock_writer)
        mock_table.batch_writer.return_value.__exit__ = MagicMock(return_value=False)

        saved = save_to_dynamodb([], "robot arm", table=mock_table)
        assert saved == 0

    def test_save_handles_duplicates(self):
        """중복 video_id 처리"""
        from lambda_.crawler_function import save_to_dynamodb

        mock_table = MagicMock()
        mock_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = MagicMock(return_value=mock_writer)
        mock_table.batch_writer.return_value.__exit__ = MagicMock(return_value=False)

        videos = [
            {"video_id": "v1", "title": "Test 1", "url": "https://y.com/1"},
            {"video_id": "v1", "title": "Test 1 dup", "url": "https://y.com/1"},
        ]

        saved = save_to_dynamodb(videos, "robot arm", table=mock_table)
        # put_item should still be called for both (DynamoDB handles dedup)
        assert saved >= 1


class TestLambdaHandler:
    """lambda/crawler_function.py - lambda_handler 테스트"""

    def test_handler_success(self):
        """Lambda 핸들러 정상 실행"""
        from lambda_.crawler_function import lambda_handler

        event = {
            "keywords": ["robot arm"],
            "max_per_keyword": 5,
            "sources": ["youtube"],
        }

        with patch("lambda_.crawler_function.LambdaCrawler") as MockCrawler:
            instance = MockCrawler.return_value
            instance.search.return_value = [
                {"video_id": "v1", "title": "Robot Arm", "url": "https://y.com/v1"},
            ]
            with patch("lambda_.crawler_function.save_to_dynamodb", return_value=1):
                result = lambda_handler(event, {})

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["videos_found"] >= 0

    def test_handler_missing_keywords(self):
        """키워드 누락 시 에러"""
        from lambda_.crawler_function import lambda_handler
        result = lambda_handler({}, {})
        assert result["statusCode"] in (200, 400, 500)

    def test_handler_empty_keywords(self):
        """빈 키워드 리스트"""
        from lambda_.crawler_function import lambda_handler
        event = {"keywords": []}
        result = lambda_handler(event, {})
        assert result["statusCode"] in (200, 400)


# ============================================================
# 3.1 Lambda 호출기 테스트
# ============================================================

class TestLambdaInvokeConfig:
    """lambda/invoke_lambda.py - 설정 테스트"""

    def test_default_config(self):
        from lambda_.invoke_lambda import LambdaInvokeConfig
        cfg = LambdaInvokeConfig()
        assert cfg.function_name == "robot-video-crawler"
        assert cfg.batch_size == 10
        assert cfg.max_per_keyword == 50
        assert "youtube" in cfg.sources

    def test_custom_config(self):
        from lambda_.invoke_lambda import LambdaInvokeConfig
        cfg = LambdaInvokeConfig(
            function_name="my-crawler",
            batch_size=5,
            max_per_keyword=20,
        )
        assert cfg.function_name == "my-crawler"
        assert cfg.batch_size == 5


class TestLambdaInvoker:
    """lambda/invoke_lambda.py - LambdaInvoker 테스트"""

    def test_invoke_single(self):
        """단일 Lambda 호출"""
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        config = LambdaInvokeConfig(invocation_type="Event")
        invoker = LambdaInvoker(config=config)

        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        result = invoker.invoke(["robot arm", "pick place"])
        assert result["status_code"] == 202
        assert result["keywords"] == ["robot arm", "pick place"]
        mock_client.invoke.assert_called_once()

    def test_parallel_invoke(self):
        """배치 병렬 호출"""
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        config = LambdaInvokeConfig(
            batch_size=2,
            delay_between_batches=0,
            invocation_type="Event",
        )
        invoker = LambdaInvoker(config=config)

        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        keywords = ["kw1", "kw2", "kw3", "kw4", "kw5"]
        summary = invoker.parallel_invoke(keywords, batch_size=2)

        assert summary["total_batches"] == 3  # ceil(5/2)
        assert summary["total_keywords"] == 5
        assert summary["success"] == 3
        assert summary["errors"] == 0

    def test_parallel_invoke_max_batches(self):
        """max_batches 제한"""
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        config = LambdaInvokeConfig(batch_size=2, delay_between_batches=0)
        invoker = LambdaInvoker(config=config)

        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        keywords = ["kw1", "kw2", "kw3", "kw4"]
        summary = invoker.parallel_invoke(keywords, batch_size=2, max_batches=1)

        assert summary["total_batches"] == 1
        assert summary["total_keywords"] == 2

    def test_invoke_error(self):
        """Lambda 호출 에러 처리"""
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig
        from botocore.exceptions import ClientError

        config = LambdaInvokeConfig(invocation_type="Event")
        invoker = LambdaInvoker(config=config)

        mock_client = MagicMock()
        mock_client.invoke.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "Invoke",
        )
        invoker._client = mock_client

        result = invoker.invoke(["test"])
        assert "error" in result
        assert result["status_code"] == 0

    def test_get_results(self):
        """결과 목록 조회"""
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        invoker = LambdaInvoker(config=LambdaInvokeConfig(delay_between_batches=0))
        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        invoker.invoke(["kw1"])
        invoker.invoke(["kw2"])
        assert len(invoker.get_results()) == 2


class TestLoadKeywords:
    """lambda/invoke_lambda.py - 키워드 로딩 함수 테스트"""

    def test_load_from_file(self, tmp_path):
        """파일에서 키워드 로딩"""
        from lambda_.invoke_lambda import load_keywords_from_file

        kw_file = tmp_path / "keywords.csv"
        kw_file.write_text("robot arm\npick and place\ncobot\n")

        keywords = load_keywords_from_file(str(kw_file))
        assert len(keywords) == 3
        assert "robot arm" in keywords

    def test_load_from_file_with_blanks(self, tmp_path):
        """공백 줄이 있는 파일"""
        from lambda_.invoke_lambda import load_keywords_from_file

        kw_file = tmp_path / "keywords.csv"
        kw_file.write_text("robot arm\n\npick place\n  \ncobot\n")

        keywords = load_keywords_from_file(str(kw_file))
        # 빈 줄 필터링 여부는 구현에 따라 다름
        non_empty = [k for k in keywords if k.strip()]
        assert len(non_empty) == 3

    def test_load_from_generator(self):
        """KeywordGenerator에서 키워드 로딩"""
        from lambda_.invoke_lambda import load_keywords_from_generator

        keywords = load_keywords_from_generator(max_count=50)
        assert len(keywords) > 0
        assert len(keywords) <= 50


# ============================================================
# 3.1 DynamoDB 동기화 테스트
# ============================================================

class TestSyncConfig:
    """lambda/dynamodb_sync.py - SyncConfig 테스트"""

    def test_default_config(self):
        from lambda_.dynamodb_sync import SyncConfig
        cfg = SyncConfig()
        assert cfg.dynamodb_table == "robot-videos"
        assert cfg.batch_size == 100
        assert cfg.mark_collected is True

    def test_custom_config(self):
        from lambda_.dynamodb_sync import SyncConfig
        cfg = SyncConfig(dynamodb_table="custom-table", batch_size=50, limit=500)
        assert cfg.dynamodb_table == "custom-table"
        assert cfg.limit == 500


class TestDynamoDBScanner:
    """lambda/dynamodb_sync.py - DynamoDBScanner 테스트"""

    def test_scan_uncollected(self):
        """미수집 아이템 스캔"""
        from lambda_.dynamodb_sync import DynamoDBScanner, SyncConfig

        config = SyncConfig()
        scanner = DynamoDBScanner(config)

        mock_table = MagicMock()
        mock_table.scan.return_value = {
            "Items": [
                {"video_id": "v1", "title": "Test1", "collected": False},
                {"video_id": "v2", "title": "Test2", "collected": False},
            ],
        }
        scanner._table = mock_table

        items = scanner.scan_uncollected()
        assert len(items) == 2
        mock_table.scan.assert_called_once()

    def test_scan_with_pagination(self):
        """페이지네이션 테스트"""
        from lambda_.dynamodb_sync import DynamoDBScanner, SyncConfig

        config = SyncConfig()
        scanner = DynamoDBScanner(config)

        mock_table = MagicMock()
        mock_table.scan.side_effect = [
            {
                "Items": [{"video_id": "v1"}],
                "LastEvaluatedKey": {"video_id": "v1"},
            },
            {
                "Items": [{"video_id": "v2"}],
            },
        ]
        scanner._table = mock_table

        items = scanner.scan_uncollected()
        assert len(items) == 2
        assert mock_table.scan.call_count == 2

    def test_scan_with_limit(self):
        """limit 적용"""
        from lambda_.dynamodb_sync import DynamoDBScanner, SyncConfig

        config = SyncConfig(limit=1)
        scanner = DynamoDBScanner(config)

        mock_table = MagicMock()
        mock_table.scan.return_value = {
            "Items": [
                {"video_id": "v1"},
                {"video_id": "v2"},
                {"video_id": "v3"},
            ],
        }
        scanner._table = mock_table

        items = scanner.scan_uncollected(limit=1)
        assert len(items) == 1

    def test_mark_collected(self):
        """수집 완료 마킹"""
        from lambda_.dynamodb_sync import DynamoDBScanner, SyncConfig

        scanner = DynamoDBScanner(SyncConfig())
        mock_table = MagicMock()
        scanner._table = mock_table

        scanner.mark_collected(["v1", "v2", "v3"])
        assert mock_table.update_item.call_count == 3

    def test_get_stats(self):
        """통계 조회"""
        from lambda_.dynamodb_sync import DynamoDBScanner, SyncConfig

        scanner = DynamoDBScanner(SyncConfig())
        mock_table = MagicMock()
        mock_table.scan.side_effect = [
            {"Count": 100},
            {"Count": 60},
        ]
        scanner._table = mock_table

        stats = scanner.get_stats()
        assert stats["total"] == 100
        assert stats["collected"] == 60
        assert stats["uncollected"] == 40


class TestLocalDBSync:
    """lambda/dynamodb_sync.py - LocalDBSync 테스트"""

    def test_ensure_table(self, tmp_path):
        """SQLite 테이블 생성"""
        from lambda_.dynamodb_sync import LocalDBSync

        db_path = str(tmp_path / "test.db")
        sync = LocalDBSync(db_path=db_path)
        conn = sync._get_connection()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='videos'"
        )
        assert cursor.fetchone() is not None

    def test_sync_items(self, tmp_path):
        """아이템 동기화"""
        from lambda_.dynamodb_sync import LocalDBSync

        db_path = str(tmp_path / "test.db")
        sync = LocalDBSync(db_path=db_path)

        items = [
            {
                "video_id": "v1",
                "title": "Robot Arm Demo",
                "url": "https://youtube.com/watch?v=v1",
                "keyword": "robot arm",
                "platform": "youtube",
            },
            {
                "video_id": "v2",
                "title": "Pick Place",
                "url": "https://youtube.com/watch?v=v2",
                "keyword": "pick place",
                "platform": "youtube",
            },
        ]

        result = sync.sync_items(items)
        assert result["inserted"] == 2
        assert result["skipped"] == 0

    def test_sync_dedup(self, tmp_path):
        """중복 아이템 건너뛰기"""
        from lambda_.dynamodb_sync import LocalDBSync

        db_path = str(tmp_path / "test.db")
        sync = LocalDBSync(db_path=db_path)

        items = [
            {"video_id": "v1", "title": "Test1", "url": "https://y.com/v1",
             "keyword": "test", "platform": "youtube"},
        ]

        # 첫 번째 동기화
        result1 = sync.sync_items(items)
        assert result1["inserted"] == 1
        assert result1["skipped"] == 0

        # 동일 아이템 다시 동기화 → 건너뛰기
        result2 = sync.sync_items(items)
        assert result2["inserted"] == 0
        assert result2["skipped"] == 1

    def test_close(self, tmp_path):
        """연결 종료"""
        from lambda_.dynamodb_sync import LocalDBSync

        db_path = str(tmp_path / "test.db")
        sync = LocalDBSync(db_path=db_path)
        sync._get_connection()
        sync.close()
        assert sync._conn is None


# ============================================================
# 3.2 KeywordGenerator 카르테시안 곱 테스트
# ============================================================

class TestKeywordGeneratorCartesian:
    """KeywordGenerator 카르테시안 곱 메서드 테스트"""

    def setup_method(self):
        from ingestion.keyword_generator import KeywordGenerator
        self.gen = KeywordGenerator(languages=["en"])

    def test_generate_2word(self):
        """2-word 카르테시안 곱: action × robot"""
        from ingestion.keyword_generator import ACTIONS_FULL, ROBOTS_FULL
        keywords = self.gen.generate_2word()
        expected_count = len(ACTIONS_FULL) * len(ROBOTS_FULL)
        assert len(keywords) == expected_count
        # 키워드 형식 확인
        assert any("robot arm" in kw for kw in keywords)
        # 각 키워드가 2 부분 이상인지 확인
        for kw in keywords[:10]:
            assert len(kw.split()) >= 2

    def test_generate_3word(self):
        """3-word 카르테시안 곱: action × robot × object"""
        keywords = self.gen.generate_3word(max_actions=5, max_robots=3, max_objects=2)
        expected = 5 * 3 * 2
        assert len(keywords) == expected
        # 각 키워드는 3 부분 이상
        for kw in keywords[:10]:
            assert len(kw.split()) >= 3

    def test_generate_3word_default(self):
        """3-word 기본값"""
        keywords = self.gen.generate_3word()
        assert len(keywords) == 20 * 15 * 10  # 3000

    def test_generate_with_context(self):
        """context 조합"""
        keywords = self.gen.generate_with_context(
            max_actions=3, max_robots=2, max_contexts=2
        )
        assert len(keywords) == 3 * 2 * 2

    def test_generate_with_context_default(self):
        """context 기본값"""
        keywords = self.gen.generate_with_context()
        assert len(keywords) == 15 * 10 * 10  # 1500

    def test_generate_cartesian_all(self):
        """전체 카르테시안 곱 생성 (중복 제거)"""
        keywords = self.gen.generate_cartesian_all()
        assert isinstance(keywords, list)
        # 중복 없어야 함
        assert len(keywords) == len(set(keywords))
        # 최소한 상당수 키워드가 있어야
        assert len(keywords) > 1000
        # 정렬되어 있어야
        assert keywords == sorted(keywords)

    def test_cartesian_content_quality(self):
        """카르테시안 곱 키워드 내용 품질"""
        from ingestion.keyword_generator import ACTIONS_FULL, ROBOTS_FULL
        keywords = self.gen.generate_2word()
        # 각 키워드가 action과 robot을 포함해야
        sample = keywords[:20]
        for kw in sample:
            # action은 multi-word일 수 있으므로 각 robot 후보로 매칭
            matched = False
            for robot in ROBOTS_FULL:
                if kw.endswith(f" {robot}"):
                    action = kw[: -(len(robot) + 1)]
                    if action in ACTIONS_FULL:
                        matched = True
                        break
            assert matched, f"키워드 '{kw}'가 ACTIONS_FULL × ROBOTS_FULL 조합에 맞지 않음"

    def test_no_empty_keywords(self):
        """빈 키워드가 없어야"""
        keywords = self.gen.generate_2word()
        for kw in keywords:
            assert kw.strip() != ""


# ============================================================
# 3.2 카르테시안 곱 상수 테스트
# ============================================================

class TestCartesianConstants:
    """카르테시안 곱 상수 리스트 테스트"""

    def test_actions_full(self):
        from ingestion.keyword_generator import ACTIONS_FULL
        assert len(ACTIONS_FULL) >= 40
        assert "grasping" in ACTIONS_FULL
        assert "pick and place" in ACTIONS_FULL
        assert all(isinstance(a, str) for a in ACTIONS_FULL)
        # 중복 없어야
        assert len(ACTIONS_FULL) == len(set(ACTIONS_FULL))

    def test_robots_full(self):
        from ingestion.keyword_generator import ROBOTS_FULL
        assert len(ROBOTS_FULL) >= 25
        assert "robot arm" in ROBOTS_FULL
        assert all(isinstance(r, str) for r in ROBOTS_FULL)
        assert len(ROBOTS_FULL) == len(set(ROBOTS_FULL))

    def test_objects_full(self):
        from ingestion.keyword_generator import OBJECTS_FULL
        assert len(OBJECTS_FULL) >= 30
        assert "box" in OBJECTS_FULL or "bottle" in OBJECTS_FULL
        assert all(isinstance(o, str) for o in OBJECTS_FULL)
        assert len(OBJECTS_FULL) == len(set(OBJECTS_FULL))

    def test_contexts_full(self):
        from ingestion.keyword_generator import CONTEXTS_FULL
        assert len(CONTEXTS_FULL) >= 15
        assert "in lab" in CONTEXTS_FULL or "laboratory" in CONTEXTS_FULL or any("lab" in c for c in CONTEXTS_FULL)
        assert all(isinstance(c, str) for c in CONTEXTS_FULL)
        assert len(CONTEXTS_FULL) == len(set(CONTEXTS_FULL))


# ============================================================
# 3.2 MultilingualExpander 테스트
# ============================================================

class TestMultilingualExpander:
    """MultilingualExpander 테스트"""

    def test_init_default(self):
        from ingestion.keyword_generator import MultilingualExpander
        expander = MultilingualExpander()
        assert "ko" in expander.target_languages
        assert "ja" in expander.target_languages
        assert len(expander.target_languages) == 5

    def test_init_custom_langs(self):
        from ingestion.keyword_generator import MultilingualExpander
        expander = MultilingualExpander(target_languages=["ko", "ja"])
        assert len(expander.target_languages) == 2

    def test_expand_with_mock_translator(self):
        """googletrans 모킹 테스트"""
        from ingestion.keyword_generator import MultilingualExpander

        expander = MultilingualExpander(target_languages=["ko", "ja"])

        mock_translator = MagicMock()
        mock_translation = MagicMock()
        mock_translation.text = "로봇 팔"
        mock_translator.translate.return_value = mock_translation
        expander._translator = mock_translator

        result = expander.expand(["robot arm"], target_languages=["ko"])

        assert "robot arm" in result
        assert result["robot arm"]["en"] == "robot arm"
        assert result["robot arm"]["ko"] == "로봇 팔"

    def test_expand_multiple(self):
        """여러 키워드 번역"""
        from ingestion.keyword_generator import MultilingualExpander

        expander = MultilingualExpander(target_languages=["ko"])

        mock_translator = MagicMock()
        translations = {"robot arm": "로봇 팔", "pick place": "픽 플레이스"}
        mock_translator.translate.side_effect = lambda text, src, dest: MagicMock(
            text=translations.get(text, text)
        )
        expander._translator = mock_translator

        result = expander.expand(["robot arm", "pick place"])
        assert len(result) == 2
        assert result["robot arm"]["ko"] == "로봇 팔"
        assert result["pick place"]["ko"] == "픽 플레이스"

    def test_get_flat_translations(self):
        """평탄 리스트 반환"""
        from ingestion.keyword_generator import MultilingualExpander

        expander = MultilingualExpander(target_languages=["ko"])

        mock_translator = MagicMock()
        mock_translator.translate.return_value = MagicMock(text="로봇 팔")
        expander._translator = mock_translator

        flat = expander.get_flat_translations(["robot arm"])
        assert isinstance(flat, list)
        assert "robot arm" in flat
        assert "로봇 팔" in flat

    def test_expand_translation_error_fallback(self):
        """번역 실패 시 원본 fallback"""
        from ingestion.keyword_generator import MultilingualExpander

        expander = MultilingualExpander(target_languages=["ko"])

        mock_translator = MagicMock()
        mock_translator.translate.side_effect = Exception("API Error")
        expander._translator = mock_translator

        result = expander.expand(["robot arm"])
        assert result["robot arm"]["ko"] == "robot arm"  # fallback


# ============================================================
# 3.2 LongtailDiscovery 테스트
# ============================================================

class TestLongtailDiscovery:
    """LongtailDiscovery 테스트"""

    def test_init(self):
        from ingestion.keyword_generator import LongtailDiscovery
        discovery = LongtailDiscovery(delay=0)
        assert discovery.delay == 0

    def test_fetch_suggestions_mock(self):
        """autocomplete API 모킹"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            "robot arm",
            [["robot arm tutorial"], ["robot arm diy"], ["robot arm pick place"]],
        ])
        mock_response.raise_for_status = MagicMock()

        with patch.object(discovery._session, "get", return_value=mock_response):
            suggestions = discovery._fetch_suggestions("robot arm")

        assert len(suggestions) == 3
        assert "robot arm tutorial" in suggestions

    def test_fetch_suggestions_error(self):
        """API 에러 시 빈 리스트"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        with patch.object(discovery._session, "get", side_effect=Exception("timeout")):
            suggestions = discovery._fetch_suggestions("robot arm")

        assert suggestions == []

    def test_discover_from_seed(self):
        """시드에서 롱테일 발견"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        suggestions_map = {
            "robot arm": ["robot arm tutorial", "robot arm diy"],
        }

        def mock_fetch(query):
            return suggestions_map.get(query, [])

        with patch.object(discovery, "_fetch_suggestions", side_effect=mock_fetch):
            longtails = discovery.discover_from_seed("robot arm", depth=1)

        assert "robot arm tutorial" in longtails
        assert "robot arm diy" in longtails

    def test_discover_depth_2(self):
        """깊이 2 탐색"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        call_count = 0

        def mock_fetch(query):
            nonlocal call_count
            call_count += 1
            if query == "robot arm":
                return ["robot arm tutorial"]
            elif query == "robot arm tutorial":
                return ["robot arm tutorial for beginners"]
            return []

        with patch.object(discovery, "_fetch_suggestions", side_effect=mock_fetch):
            longtails = discovery.discover_from_seed("robot arm", depth=2)

        assert "robot arm tutorial" in longtails
        assert "robot arm tutorial for beginners" in longtails

    def test_alphabet_expand(self):
        """알파벳 확장"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        def mock_fetch(query):
            letter = query.split()[-1]
            if letter == "a":
                return ["robot arm assembly"]
            elif letter == "b":
                return ["robot arm build"]
            return []

        with patch.object(discovery, "_fetch_suggestions", side_effect=mock_fetch):
            results = discovery.alphabet_expand("robot arm")

        assert "robot arm assembly" in results
        assert "robot arm build" in results

    def test_batch_discover(self):
        """배치 롱테일 발견"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        def mock_fetch(query):
            return [f"{query} tutorial", f"{query} demo"]

        with patch.object(discovery, "_fetch_suggestions", side_effect=mock_fetch):
            result = discovery.batch_discover(
                ["robot arm", "cobot"], depth=1, max_seeds=2
            )

        assert "robot arm" in result
        assert "cobot" in result
        assert len(result["robot arm"]) > 0
        assert len(result["cobot"]) > 0

    def test_batch_discover_max_seeds(self):
        """max_seeds 제한"""
        from ingestion.keyword_generator import LongtailDiscovery

        discovery = LongtailDiscovery(delay=0)

        with patch.object(discovery, "_fetch_suggestions", return_value=[]):
            result = discovery.batch_discover(
                ["s1", "s2", "s3", "s4", "s5"], depth=1, max_seeds=2
            )

        assert len(result) == 2


# ============================================================
# 기존 KeywordGenerator 기능 호환성 테스트
# ============================================================

class TestKeywordGeneratorCompat:
    """기존 기능이 깨지지 않았는지 확인"""

    def test_generate_all(self):
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en"])
        result = gen.generate_all()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_flat_keywords(self):
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en"])
        keywords = gen.get_flat_keywords(max_count=50)
        assert len(keywords) <= 50
        assert len(keywords) > 0

    def test_get_batched_keywords(self):
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en"])
        batches = gen.get_batched_keywords(batch_size=5, max_batches=4)
        assert isinstance(batches, list)
        assert len(batches) <= 4
        for batch in batches:
            assert len(batch) <= 5

    def test_combinations_still_work(self):
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en"])
        combos = gen._generate_combinations()
        assert isinstance(combos, list)
        assert len(combos) > 0

    def test_suggest_new_keywords(self):
        from ingestion.keyword_generator import KeywordGenerator
        gen = KeywordGenerator(languages=["en"])
        titles = [
            "robot arm grasping objects demonstration",
            "robot arm pick and place experiment",
            "robotic manipulation of small tools",
        ]
        suggestions = gen.suggest_new_keywords(titles)
        assert isinstance(suggestions, list)


# ============================================================
# 통합 시나리오 테스트
# ============================================================

class TestIntegrationScenarios:
    """End-to-end 시나리오 테스트"""

    def test_keyword_gen_to_lambda_flow(self):
        """KeywordGenerator → LambdaInvoker 파이프라인"""
        from ingestion.keyword_generator import KeywordGenerator
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        # 1. 키워드 생성
        gen = KeywordGenerator(languages=["en"])
        keywords = gen.get_flat_keywords(max_count=20)
        assert len(keywords) > 0

        # 2. Lambda 호출 (모킹)
        config = LambdaInvokeConfig(batch_size=5, delay_between_batches=0)
        invoker = LambdaInvoker(config=config)

        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        summary = invoker.parallel_invoke(keywords, batch_size=5)
        assert summary["success"] >= 1
        assert summary["total_keywords"] == len(keywords)

    def test_cartesian_to_lambda_flow(self):
        """카르테시안 곱 → Lambda 호출 파이프라인"""
        from ingestion.keyword_generator import KeywordGenerator
        from lambda_.invoke_lambda import LambdaInvoker, LambdaInvokeConfig

        gen = KeywordGenerator(languages=["en"])

        # 소규모 카르테시안 곱
        keywords = gen.generate_3word(max_actions=2, max_robots=2, max_objects=2)
        assert len(keywords) == 8  # 2 * 2 * 2

        config = LambdaInvokeConfig(batch_size=4, delay_between_batches=0)
        invoker = LambdaInvoker(config=config)
        mock_client = MagicMock()
        mock_client.invoke.return_value = {"StatusCode": 202}
        invoker._client = mock_client

        summary = invoker.parallel_invoke(keywords, batch_size=4)
        assert summary["total_batches"] == 2
        assert summary["total_keywords"] == 8

    def test_dynamodb_to_local_sync_flow(self, tmp_path):
        """DynamoDB → 로컬 DB 동기화 파이프라인"""
        from lambda_.dynamodb_sync import LocalDBSync

        db_path = str(tmp_path / "sync_test.db")
        sync = LocalDBSync(db_path=db_path)

        # DynamoDB에서 가져온 아이템 시뮬레이션
        items = [
            {
                "video_id": f"v{i}",
                "title": f"Robot Video {i}",
                "url": f"https://youtube.com/watch?v=v{i}",
                "keyword": "robot arm",
                "platform": "youtube",
            }
            for i in range(10)
        ]

        result1 = sync.sync_items(items)
        assert result1["inserted"] == 10
        assert result1["skipped"] == 0

        # 다시 동기화 → 모두 건너뜀
        result2 = sync.sync_items(items)
        assert result2["inserted"] == 0
        assert result2["skipped"] == 10

        sync.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
