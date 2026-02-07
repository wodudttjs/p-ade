"""
KeywordManager 테스트
"""

import pytest
from datetime import datetime

from ingestion.keyword_manager import KeywordManager
from models.database import Keyword, KeywordCategory, KeywordPerformance


# ============ Category Tests ============

def test_keyword_manager_import():
    """KeywordManager import 확인"""
    from ingestion.keyword_manager import KeywordManager
    assert KeywordManager is not None


def test_create_category(test_db):
    """카테고리 생성 테스트"""
    manager = KeywordManager(test_db)
    
    category = manager.create_category(
        name="Assembly",
        description="Robot assembly tasks"
    )
    
    assert category.id is not None
    assert category.name == "Assembly"
    assert category.description == "Robot assembly tasks"
    assert category.created_at is not None


def test_get_category(test_db):
    """카테고리 조회 테스트"""
    manager = KeywordManager(test_db)
    
    # 생성
    created = manager.create_category(name="Cooking")
    
    # 조회
    retrieved = manager.get_category(created.id)
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Cooking"


def test_get_category_by_name(test_db):
    """이름으로 카테고리 조회 테스트"""
    manager = KeywordManager(test_db)
    
    manager.create_category(name="Exercise")
    
    retrieved = manager.get_category_by_name("Exercise")
    assert retrieved is not None
    assert retrieved.name == "Exercise"


def test_list_categories(test_db):
    """카테고리 목록 조회 테스트"""
    manager = KeywordManager(test_db)
    
    manager.create_category(name="Category1")
    manager.create_category(name="Category2")
    manager.create_category(name="Category3")
    
    categories = manager.list_categories()
    assert len(categories) == 3
    assert any(c.name == "Category1" for c in categories)


# ============ Keyword Tests ============

def test_create_keyword(test_db):
    """키워드 생성 테스트"""
    manager = KeywordManager(test_db)
    
    category = manager.create_category(name="Assembly")
    
    keyword = manager.create_keyword(
        keyword="robot assembly",
        category_id=category.id,
        language="en",
        priority=8,
        weight=1.5
    )
    
    assert keyword.id is not None
    assert keyword.keyword == "robot assembly"
    assert keyword.category_id == category.id
    assert keyword.language == "en"
    assert keyword.priority == 8
    assert keyword.weight == 1.5
    assert keyword.is_active is True
    assert keyword.performance is not None


def test_get_keyword(test_db):
    """키워드 조회 테스트"""
    manager = KeywordManager(test_db)
    
    created = manager.create_keyword(keyword="pick and place")
    
    retrieved = manager.get_keyword(created.id)
    assert retrieved is not None
    assert retrieved.keyword == "pick and place"


def test_get_keyword_by_text(test_db):
    """텍스트로 키워드 조회 테스트"""
    manager = KeywordManager(test_db)
    
    manager.create_keyword(keyword="assembly tutorial")
    
    retrieved = manager.get_keyword_by_text("assembly tutorial")
    assert retrieved is not None
    assert retrieved.keyword == "assembly tutorial"


def test_update_keyword(test_db):
    """키워드 업데이트 테스트"""
    manager = KeywordManager(test_db)
    
    keyword = manager.create_keyword(
        keyword="robot task",
        priority=5,
        weight=1.0
    )
    
    updated = manager.update_keyword(
        keyword.id,
        priority=9,
        weight=2.0,
        is_active=False
    )
    
    assert updated is not None
    assert updated.priority == 9
    assert updated.weight == 2.0
    assert updated.is_active is False


def test_delete_keyword(test_db):
    """키워드 삭제 테스트"""
    manager = KeywordManager(test_db)
    
    keyword = manager.create_keyword(keyword="temporary keyword")
    keyword_id = keyword.id
    
    # 삭제
    result = manager.delete_keyword(keyword_id)
    assert result is True
    
    # 확인
    retrieved = manager.get_keyword(keyword_id)
    assert retrieved is None


def test_list_keywords_no_filter(test_db):
    """키워드 목록 조회 테스트 (필터 없음)"""
    manager = KeywordManager(test_db)
    
    manager.create_keyword(keyword="keyword1", priority=5)
    manager.create_keyword(keyword="keyword2", priority=8)
    manager.create_keyword(keyword="keyword3", priority=3)
    
    keywords = manager.list_keywords()
    
    # 우선순위 내림차순으로 정렬되어야 함
    assert len(keywords) == 3
    assert keywords[0].priority == 8
    assert keywords[1].priority == 5
    assert keywords[2].priority == 3


def test_list_keywords_with_filters(test_db):
    """키워드 목록 조회 테스트 (필터 적용)"""
    manager = KeywordManager(test_db)
    
    category = manager.create_category(name="TestCategory")
    
    manager.create_keyword(
        keyword="kw1",
        category_id=category.id,
        language="en",
        priority=7,
        is_active=True
    )
    manager.create_keyword(
        keyword="kw2",
        category_id=category.id,
        language="ko",
        priority=5,
        is_active=True
    )
    manager.create_keyword(
        keyword="kw3",
        language="en",
        priority=3,
        is_active=False
    )
    
    # 카테고리 필터
    result = manager.list_keywords(category_id=category.id)
    assert len(result) == 2
    
    # 언어 필터
    result = manager.list_keywords(language="ko")
    assert len(result) == 1
    assert result[0].keyword == "kw2"
    
    # 활성 상태 필터
    result = manager.list_keywords(is_active=False)
    assert len(result) == 1
    
    # 최소 우선순위 필터
    result = manager.list_keywords(min_priority=6)
    assert len(result) == 1
    assert result[0].priority == 7


def test_mark_keyword_used(test_db):
    """키워드 사용 표시 테스트"""
    manager = KeywordManager(test_db)
    
    keyword = manager.create_keyword(keyword="test keyword")
    assert keyword.last_used_at is None
    
    manager.mark_keyword_used(keyword.id)
    
    updated = manager.get_keyword(keyword.id)
    assert updated.last_used_at is not None


# ============ Query Builder Tests ============

def test_build_search_query_youtube_and(test_db):
    """YouTube AND 쿼리 생성 테스트"""
    manager = KeywordManager(test_db)
    
    query = manager.build_search_query(
        keywords=["robot", "assembly", "tutorial"],
        operator="AND",
        platform="youtube"
    )
    
    assert query == "robot assembly tutorial"


def test_build_search_query_youtube_or(test_db):
    """YouTube OR 쿼리 생성 테스트"""
    manager = KeywordManager(test_db)
    
    query = manager.build_search_query(
        keywords=["robot", "assembly"],
        operator="OR",
        platform="youtube"
    )
    
    assert query == "robot OR assembly"


def test_build_search_query_empty(test_db):
    """빈 키워드로 쿼리 생성 테스트"""
    manager = KeywordManager(test_db)
    
    query = manager.build_search_query(
        keywords=[],
        operator="AND",
        platform="youtube"
    )
    
    assert query == ""


def test_get_next_keywords(test_db):
    """다음 크롤링 키워드 선택 테스트"""
    manager = KeywordManager(test_db)
    
    manager.create_keyword(keyword="kw1", priority=9, is_active=True)
    manager.create_keyword(keyword="kw2", priority=7, is_active=True)
    manager.create_keyword(keyword="kw3", priority=5, is_active=False)
    manager.create_keyword(keyword="kw4", priority=3, is_active=True)
    
    # 활성 키워드만 선택, 우선순위 순
    keywords = manager.get_next_keywords(count=2)
    
    assert len(keywords) == 2
    assert keywords[0].priority == 9
    assert keywords[1].priority == 7


# ============ Performance Tests ============

def test_update_performance(test_db):
    """성능 지표 업데이트 테스트"""
    manager = KeywordManager(test_db)
    
    keyword = manager.create_keyword(keyword="test keyword")
    
    # 성능 업데이트
    manager.update_performance(
        keyword.id,
        videos_found=10,
        videos_downloaded=8,
        high_quality_episodes=6
    )
    
    # 확인
    updated = manager.get_keyword(keyword.id)
    perf = updated.performance
    
    assert perf.total_searches == 1
    assert perf.total_videos_found == 10
    assert perf.total_videos_downloaded == 8
    assert perf.total_high_quality_episodes == 6
    assert perf.videos_per_search == 10.0
    assert perf.success_rate == 0.8
    assert perf.quality_episodes_per_video == 0.75
    assert perf.last_calculated_at is not None


def test_update_performance_multiple_times(test_db):
    """여러 번 성능 업데이트 테스트"""
    manager = KeywordManager(test_db)
    
    keyword = manager.create_keyword(keyword="test keyword")
    
    # 첫 번째 업데이트
    manager.update_performance(
        keyword.id,
        videos_found=10,
        videos_downloaded=8
    )
    
    # 두 번째 업데이트
    manager.update_performance(
        keyword.id,
        videos_found=5,
        videos_downloaded=4
    )
    
    updated = manager.get_keyword(keyword.id)
    perf = updated.performance
    
    assert perf.total_searches == 2
    assert perf.total_videos_found == 15
    assert perf.total_videos_downloaded == 12
    assert perf.videos_per_search == 7.5  # 15 / 2


def test_get_top_performing_keywords(test_db):
    """성능 좋은 키워드 조회 테스트"""
    manager = KeywordManager(test_db)
    
    # 여러 키워드 생성 및 성능 설정
    kw1 = manager.create_keyword(keyword="kw1")
    manager.update_performance(kw1.id, videos_found=10, videos_downloaded=8)
    
    kw2 = manager.create_keyword(keyword="kw2")
    manager.update_performance(kw2.id, videos_found=10, videos_downloaded=9)
    
    kw3 = manager.create_keyword(keyword="kw3")
    manager.update_performance(kw3.id, videos_found=10, videos_downloaded=5)
    
    # success_rate 기준으로 조회
    top_keywords = manager.get_top_performing_keywords(
        limit=2,
        metric='success_rate'
    )
    
    assert len(top_keywords) == 2
    assert top_keywords[0].keyword == "kw2"  # 0.9 success rate
    assert top_keywords[1].keyword == "kw1"  # 0.8 success rate
