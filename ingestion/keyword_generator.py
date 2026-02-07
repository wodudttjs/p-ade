"""
키워드 자동 확장/생성기

로봇팔 관련 영상 대량 수집을 위한 키워드를 자동으로 생성하고 확장합니다.

기능:
- 기본 시드 키워드에서 조합 키워드 자동 생성
- 다국어 키워드 확장 (영어, 한국어, 일본어, 중국어, 독일어)
- 카테고리별 키워드 관리
- 키워드 성능 기반 우선순위 자동 조정
- 중복 제거 및 관련성 필터링
"""

import itertools
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class KeywordSet:
    """키워드 세트"""
    keywords: List[str]
    category: str
    language: str = "en"
    priority: int = 5  # 1-10
    expected_yield: int = 0  # 예상 결과 수
    last_used: Optional[datetime] = None
    total_found: int = 0
    success_rate: float = 0.0


# ============================================================
# 시드 키워드 사전
# ============================================================

SEED_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    # === 핵심 로봇팔 키워드 ===
    "robot_arm_core": {
        "en": [
            "robot arm",
            "robotic arm",
            "robot manipulator",
            "industrial robot arm",
            "collaborative robot arm",
            "cobot arm",
            "6-axis robot arm",
            "7-axis robot arm",
            "articulated robot",
            "SCARA robot",
            "delta robot",
            "cartesian robot",
        ],
        "ko": [
            "로봇팔",
            "산업용 로봇팔",
            "협동 로봇",
            "다관절 로봇",
            "로봇 매니퓰레이터",
            "로봇 그리퍼",
        ],
        "ja": [
            "ロボットアーム",
            "産業用ロボット",
            "協働ロボット",
            "多関節ロボット",
        ],
        "zh": [
            "机器人手臂",
            "工业机器人",
            "协作机器人",
            "机械臂",
        ],
        "de": [
            "Roboterarm",
            "Industrieroboter",
            "Kollaborativer Roboter",
        ],
    },

    # === 동작/태스크 키워드 ===
    "robot_actions": {
        "en": [
            "pick and place robot",
            "robot grasping",
            "robot gripper",
            "robot manipulation",
            "object manipulation robot",
            "bin picking robot",
            "robot assembly",
            "robot welding",
            "robot palletizing",
            "robot sorting",
            "robot stacking",
            "robot packaging",
            "robot loading unloading",
            "robot material handling",
            "robotic deburring",
            "robot polishing",
            "robot inspection",
        ],
        "ko": [
            "로봇 피킹",
            "로봇 용접",
            "로봇 팔레타이징",
            "로봇 조립",
            "로봇 검사",
        ],
    },

    # === 제조사/브랜드 키워드 ===
    "robot_brands": {
        "en": [
            "FANUC robot arm",
            "ABB robot arm",
            "KUKA robot arm",
            "Universal Robots UR5",
            "Universal Robots UR10",
            "Universal Robots UR3",
            "Yaskawa Motoman",
            "Kawasaki robot arm",
            "Doosan robotics",
            "Techman Robot",
            "Franka Emika Panda",
            "Kinova robot arm",
            "Denso robot arm",
            "Epson robot arm",
            "Staubli robot arm",
            "Mitsubishi robot arm",
            "Nachi robot arm",
            "Omron robot arm",
            "Aubo robot",
            "Flexiv Rizon",
        ],
    },

    # === 연구/데모 키워드 ===
    "research_demo": {
        "en": [
            "robot arm demonstration",
            "robot arm demo",
            "robot arm tutorial",
            "robot arm programming",
            "robot arm simulation real",
            "robot arm real world",
            "robot arm pick place demo",
            "robot arm grasping demo",
            "robot arm object detection",
            "robot arm vision system",
            "robot arm deep learning",
            "robot arm reinforcement learning",
            "robot arm imitation learning",
            "robot arm teleoperation",
            "robot arm dexterous manipulation",
        ],
    },

    # === 산업 응용 키워드 ===
    "industrial_applications": {
        "en": [
            "factory robot arm",
            "manufacturing robot arm",
            "production line robot",
            "CNC robot loading",
            "injection molding robot",
            "food handling robot",
            "pharmaceutical robot",
            "electronics assembly robot",
            "automotive robot arm",
            "warehouse robot arm",
            "logistics robot arm",
        ],
    },

    # === 비전/센서 키워드 ===
    "vision_sensor": {
        "en": [
            "robot arm camera",
            "robot arm depth sensor",
            "robot arm 3D vision",
            "robot arm force sensor",
            "robot hand eye coordination",
            "visual servoing robot",
            "robot arm point cloud",
            "robot arm pose estimation",
        ],
    },
}

# === 동작 수식어 (조합용) ===
ACTION_MODIFIERS = [
    "real time", "high speed", "precise", "automated",
    "continuous", "repetitive", "smooth", "fast",
]

# === 환경 수식어 (조합용) ===
ENVIRONMENT_MODIFIERS = [
    "factory", "lab", "laboratory", "production",
    "warehouse", "cleanroom", "workshop",
]

# === 제외 키워드 ===
EXCLUDE_TERMS = {
    "animation", "simulation", "cgi", "3d render", "3d model",
    "toy", "lego", "game", "surgery", "medical", "prosthetic",
    "drawing", "cartoon", "unboxing", "review", "price",
    "how much", "buy", "amazon", "aliexpress",
}


class KeywordGenerator:
    """키워드 자동 생성기"""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_keywords: int = 500,
    ):
        """
        Args:
            languages: 사용할 언어 목록 (기본: ["en"])
            categories: 사용할 카테고리 목록 (기본: 전체)
            max_keywords: 최대 키워드 수
        """
        self.languages = languages or ["en"]
        self.categories = categories
        self.max_keywords = max_keywords
        self._used_keywords: Set[str] = set()
        self._keyword_stats: Dict[str, Dict] = {}

    def generate_all(self) -> List[KeywordSet]:
        """모든 카테고리에서 키워드 세트 생성"""
        all_sets: List[KeywordSet] = []

        for cat_name, lang_dict in SEED_KEYWORDS.items():
            if self.categories and cat_name not in self.categories:
                continue

            for lang, keywords in lang_dict.items():
                if lang not in self.languages:
                    continue

                ks = KeywordSet(
                    keywords=keywords,
                    category=cat_name,
                    language=lang,
                    priority=self._category_priority(cat_name),
                )
                all_sets.append(ks)

        # 조합 키워드 추가
        if "en" in self.languages:
            combo_sets = self._generate_combinations()
            all_sets.extend(combo_sets)

        logger.info(
            f"생성된 키워드 세트: {len(all_sets)}개, "
            f"총 키워드: {sum(len(s.keywords) for s in all_sets)}개"
        )
        return all_sets

    def get_flat_keywords(
        self,
        max_count: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[str]:
        """모든 키워드를 평탄화하여 반환 (중복 제거)"""
        all_sets = self.generate_all()
        seen: Set[str] = set()
        result: List[str] = []

        # 우선순위별 정렬 (높은 것 먼저)
        all_sets.sort(key=lambda s: s.priority, reverse=True)

        for ks in all_sets:
            for kw in ks.keywords:
                kw_lower = kw.lower().strip()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    result.append(kw)

        if shuffle:
            import random
            # 우선순위 그룹 내에서만 셔플
            random.shuffle(result)

        limit = max_count or self.max_keywords
        return result[:limit]

    def get_batched_keywords(
        self,
        batch_size: int = 10,
        max_batches: Optional[int] = None,
    ) -> List[List[str]]:
        """키워드를 배치 단위로 반환"""
        flat = self.get_flat_keywords()
        batches = []
        for i in range(0, len(flat), batch_size):
            batches.append(flat[i:i + batch_size])
            if max_batches and len(batches) >= max_batches:
                break
        return batches

    def _generate_combinations(self) -> List[KeywordSet]:
        """수식어 조합 키워드 생성"""
        base_terms = [
            "robot arm", "robotic arm", "robot gripper",
            "pick and place", "robot manipulation",
        ]
        combos: List[str] = []

        # 동작 수식어 조합
        for base, mod in itertools.product(base_terms[:3], ACTION_MODIFIERS[:4]):
            combos.append(f"{mod} {base}")

        # 환경 수식어 조합
        for base, env in itertools.product(base_terms[:3], ENVIRONMENT_MODIFIERS[:4]):
            combos.append(f"{base} {env}")

        return [
            KeywordSet(
                keywords=combos,
                category="combinations",
                language="en",
                priority=4,
            )
        ]

    def _category_priority(self, category: str) -> int:
        """카테고리별 기본 우선순위"""
        priorities = {
            "robot_arm_core": 10,
            "robot_actions": 9,
            "robot_brands": 7,
            "research_demo": 8,
            "industrial_applications": 6,
            "vision_sensor": 5,
            "combinations": 4,
        }
        return priorities.get(category, 5)

    def update_stats(
        self,
        keyword: str,
        found_count: int,
        downloaded_count: int = 0,
        quality_count: int = 0,
    ):
        """키워드 성능 통계 업데이트"""
        if keyword not in self._keyword_stats:
            self._keyword_stats[keyword] = {
                "total_found": 0,
                "total_downloaded": 0,
                "total_quality": 0,
                "search_count": 0,
            }
        stats = self._keyword_stats[keyword]
        stats["total_found"] += found_count
        stats["total_downloaded"] += downloaded_count
        stats["total_quality"] += quality_count
        stats["search_count"] += 1

    def get_top_keywords(self, n: int = 20) -> List[Tuple[str, Dict]]:
        """성능 상위 키워드 반환"""
        sorted_kw = sorted(
            self._keyword_stats.items(),
            key=lambda x: x[1].get("total_found", 0),
            reverse=True,
        )
        return sorted_kw[:n]

    def suggest_new_keywords(self, existing_titles: List[str]) -> List[str]:
        """기존 수집 영상 제목에서 새 키워드 추출 (간단한 TF 기반)"""
        from collections import Counter

        # 단어 빈도 계산
        word_freq = Counter()
        for title in existing_titles:
            words = title.lower().split()
            for w in words:
                w = w.strip(".,!?()[]{}\"'")
                if len(w) > 3 and w not in EXCLUDE_TERMS:
                    word_freq[w] += 1

        # 로봇 관련 용어와 동시 출현하는 단어 추출
        robot_terms = {"robot", "arm", "robotic", "gripper", "manipulator"}
        suggestions = []
        for word, count in word_freq.most_common(50):
            if word not in robot_terms and count >= 2:
                # 로봇 용어와 조합
                suggestions.append(f"robot arm {word}")
                suggestions.append(f"robotic {word}")

        return suggestions[:20]


# CLI 진입점
if __name__ == "__main__":
    gen = KeywordGenerator(languages=["en", "ko"])
    keywords = gen.get_flat_keywords(max_count=100)
    print(f"총 {len(keywords)}개 키워드 생성:")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:3d}. {kw}")
