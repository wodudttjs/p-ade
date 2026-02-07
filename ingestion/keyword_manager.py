"""
키워드 관리 시스템

검색 키워드의 CRUD 연산 및 쿼리 빌더 기능을 제공합니다.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from models.database import Keyword, KeywordCategory, KeywordPerformance
from core.logging_config import logger


class KeywordManager:
    """키워드 관리 클래스"""
    
    def __init__(self, db_session: Session):
        """
        Args:
            db_session: SQLAlchemy 데이터베이스 세션
        """
        self.db = db_session
    
    # ============ Category Management ============
    
    def create_category(
        self, 
        name: str, 
        description: Optional[str] = None
    ) -> KeywordCategory:
        """
        키워드 카테고리 생성
        
        Args:
            name: 카테고리 이름
            description: 카테고리 설명
            
        Returns:
            생성된 KeywordCategory 객체
        """
        category = KeywordCategory(
            name=name,
            description=description
        )
        self.db.add(category)
        self.db.commit()
        self.db.refresh(category)
        
        logger.info(f"Created category: {name} (ID: {category.id})")
        return category
    
    def get_category(self, category_id: int) -> Optional[KeywordCategory]:
        """
        ID로 카테고리 조회
        
        Args:
            category_id: 카테고리 ID
            
        Returns:
            KeywordCategory 객체 또는 None
        """
        return self.db.query(KeywordCategory).filter(
            KeywordCategory.id == category_id
        ).first()
    
    def get_category_by_name(self, name: str) -> Optional[KeywordCategory]:
        """
        이름으로 카테고리 조회
        
        Args:
            name: 카테고리 이름
            
        Returns:
            KeywordCategory 객체 또는 None
        """
        return self.db.query(KeywordCategory).filter(
            KeywordCategory.name == name
        ).first()
    
    def list_categories(self) -> List[KeywordCategory]:
        """
        모든 카테고리 조회
        
        Returns:
            KeywordCategory 객체 리스트
        """
        return self.db.query(KeywordCategory).all()
    
    # ============ Keyword Management ============
    
    def create_keyword(
        self,
        keyword: str,
        category_id: Optional[int] = None,
        language: str = 'en',
        priority: int = 5,
        weight: float = 1.0,
        is_active: bool = True
    ) -> Keyword:
        """
        키워드 생성
        
        Args:
            keyword: 키워드 텍스트
            category_id: 카테고리 ID (선택)
            language: 언어 코드 (기본값: 'en')
            priority: 우선순위 1-10 (기본값: 5)
            weight: 가중치 (기본값: 1.0)
            is_active: 활성 상태 (기본값: True)
            
        Returns:
            생성된 Keyword 객체
        """
        kw = Keyword(
            keyword=keyword,
            category_id=category_id,
            language=language,
            priority=priority,
            weight=weight,
            is_active=is_active
        )
        self.db.add(kw)
        self.db.commit()
        self.db.refresh(kw)
        
        # 성능 지표 초기화
        performance = KeywordPerformance(keyword_id=kw.id)
        self.db.add(performance)
        self.db.commit()
        
        logger.info(f"Created keyword: {keyword} (ID: {kw.id})")
        return kw
    
    def get_keyword(self, keyword_id: int) -> Optional[Keyword]:
        """
        ID로 키워드 조회
        
        Args:
            keyword_id: 키워드 ID
            
        Returns:
            Keyword 객체 또는 None
        """
        return self.db.query(Keyword).filter(
            Keyword.id == keyword_id
        ).first()
    
    def get_keyword_by_text(self, keyword: str) -> Optional[Keyword]:
        """
        텍스트로 키워드 조회
        
        Args:
            keyword: 키워드 텍스트
            
        Returns:
            Keyword 객체 또는 None
        """
        return self.db.query(Keyword).filter(
            Keyword.keyword == keyword
        ).first()
    
    def update_keyword(
        self,
        keyword_id: int,
        **kwargs
    ) -> Optional[Keyword]:
        """
        키워드 업데이트
        
        Args:
            keyword_id: 키워드 ID
            **kwargs: 업데이트할 필드들
            
        Returns:
            업데이트된 Keyword 객체 또는 None
        """
        keyword = self.get_keyword(keyword_id)
        if not keyword:
            logger.warning(f"Keyword ID {keyword_id} not found")
            return None
        
        for key, value in kwargs.items():
            if hasattr(keyword, key):
                setattr(keyword, key, value)
        
        self.db.commit()
        self.db.refresh(keyword)
        
        logger.info(f"Updated keyword ID {keyword_id}: {kwargs}")
        return keyword
    
    def delete_keyword(self, keyword_id: int) -> bool:
        """
        키워드 삭제
        
        Args:
            keyword_id: 키워드 ID
            
        Returns:
            삭제 성공 여부
        """
        keyword = self.get_keyword(keyword_id)
        if not keyword:
            logger.warning(f"Keyword ID {keyword_id} not found")
            return False
        
        # 성능 지표도 함께 삭제
        if keyword.performance:
            self.db.delete(keyword.performance)
        
        self.db.delete(keyword)
        self.db.commit()
        
        logger.info(f"Deleted keyword ID {keyword_id}")
        return True
    
    def list_keywords(
        self,
        category_id: Optional[int] = None,
        language: Optional[str] = None,
        is_active: Optional[bool] = None,
        min_priority: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Keyword]:
        """
        키워드 목록 조회 (필터링 지원)
        
        Args:
            category_id: 카테고리 ID 필터
            language: 언어 필터
            is_active: 활성 상태 필터
            min_priority: 최소 우선순위 필터
            limit: 최대 결과 수
            
        Returns:
            Keyword 객체 리스트
        """
        query = self.db.query(Keyword)
        
        # 필터 적용
        if category_id is not None:
            query = query.filter(Keyword.category_id == category_id)
        if language is not None:
            query = query.filter(Keyword.language == language)
        if is_active is not None:
            query = query.filter(Keyword.is_active == is_active)
        if min_priority is not None:
            query = query.filter(Keyword.priority >= min_priority)
        
        # 우선순위 내림차순 정렬
        query = query.order_by(desc(Keyword.priority))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def mark_keyword_used(self, keyword_id: int):
        """
        키워드 사용 시간 업데이트
        
        Args:
            keyword_id: 키워드 ID
        """
        keyword = self.get_keyword(keyword_id)
        if keyword:
            keyword.last_used_at = datetime.utcnow()
            self.db.commit()
    
    # ============ Query Builder ============
    
    def build_search_query(
        self,
        keywords: List[str],
        operator: str = 'AND',
        platform: str = 'youtube'
    ) -> str:
        """
        검색 쿼리 생성
        
        Args:
            keywords: 키워드 리스트
            operator: 연산자 ('AND' 또는 'OR')
            platform: 플랫폼 이름
            
        Returns:
            플랫폼에 맞는 검색 쿼리 문자열
        """
        if not keywords:
            return ""
        
        if platform.lower() == 'youtube':
            # YouTube 검색 형식
            if operator.upper() == 'OR':
                return ' OR '.join(keywords)
            else:  # AND
                return ' '.join(keywords)
        
        elif platform.lower() == 'vimeo':
            # Vimeo 검색 형식
            if operator.upper() == 'OR':
                return ' OR '.join(keywords)
            else:
                return ' '.join(keywords)
        
        else:
            # 기본 형식 (공백으로 구분)
            return ' '.join(keywords)
    
    def get_next_keywords(
        self,
        count: int = 10,
        language: Optional[str] = None,
        category_id: Optional[int] = None
    ) -> List[Keyword]:
        """
        다음 크롤링에 사용할 키워드 선택
        
        우선순위와 가중치를 고려하여 키워드를 선택합니다.
        
        Args:
            count: 선택할 키워드 수
            language: 언어 필터 (선택)
            category_id: 카테고리 필터 (선택)
            
        Returns:
            선택된 Keyword 객체 리스트
        """
        keywords = self.list_keywords(
            category_id=category_id,
            language=language,
            is_active=True,
            min_priority=1,
            limit=count
        )
        
        logger.info(f"Selected {len(keywords)} keywords for next crawl")
        return keywords
    
    # ============ Performance Tracking ============
    
    def update_performance(
        self,
        keyword_id: int,
        videos_found: int = 0,
        videos_downloaded: int = 0,
        high_quality_episodes: int = 0
    ):
        """
        키워드 성능 지표 업데이트
        
        Args:
            keyword_id: 키워드 ID
            videos_found: 발견된 비디오 수
            videos_downloaded: 다운로드된 비디오 수
            high_quality_episodes: 고품질 에피소드 수
        """
        keyword = self.get_keyword(keyword_id)
        if not keyword or not keyword.performance:
            logger.warning(f"Keyword or performance not found for ID {keyword_id}")
            return
        
        perf = keyword.performance
        perf.total_searches += 1
        perf.total_videos_found += videos_found
        perf.total_videos_downloaded += videos_downloaded
        perf.total_high_quality_episodes += high_quality_episodes
        
        # 평균 계산
        if perf.total_searches > 0:
            perf.videos_per_search = perf.total_videos_found / perf.total_searches
        
        if perf.total_videos_downloaded > 0:
            perf.success_rate = perf.total_videos_downloaded / perf.total_videos_found if perf.total_videos_found > 0 else 0
            perf.quality_episodes_per_video = perf.total_high_quality_episodes / perf.total_videos_downloaded
        
        perf.last_calculated_at = datetime.utcnow()
        
        self.db.commit()
        logger.info(f"Updated performance for keyword ID {keyword_id}")
    
    def get_top_performing_keywords(
        self,
        limit: int = 10,
        metric: str = 'success_rate'
    ) -> List[Keyword]:
        """
        성능이 좋은 키워드 조회
        
        Args:
            limit: 최대 결과 수
            metric: 정렬 기준 ('success_rate', 'videos_per_search', etc.)
            
        Returns:
            성능 순으로 정렬된 Keyword 객체 리스트
        """
        query = self.db.query(Keyword).join(KeywordPerformance)
        
        # 정렬 기준 적용
        if metric == 'success_rate':
            query = query.order_by(desc(KeywordPerformance.success_rate))
        elif metric == 'videos_per_search':
            query = query.order_by(desc(KeywordPerformance.videos_per_search))
        elif metric == 'quality_episodes_per_video':
            query = query.order_by(desc(KeywordPerformance.quality_episodes_per_video))
        else:
            query = query.order_by(desc(KeywordPerformance.total_videos_found))
        
        return query.limit(limit).all()
