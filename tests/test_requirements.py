"""
requirements.txt 테스트

requirements.txt 파일의 내용을 검증합니다.
"""

import pytest
from pathlib import Path


def test_requirements_file_exists():
    """requirements.txt 파일 존재 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    assert requirements_file.exists()


def test_requirements_syntax():
    """requirements.txt 구문 검증"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text()
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
    
    for line in lines:
        # 각 줄이 유효한 패키지 명세인지 확인
        assert '=' in line or '>=' in line or '<' in line or line.replace('-', '').replace('_', '').isalnum(), \
            f"유효하지 않은 패키지 명세: {line}"


def test_essential_packages_included():
    """필수 패키지 포함 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text().lower()
    
    essential_packages = {
        # 웹 스크래핑
        'scrapy': '웹 크롤링 프레임워크',
        'playwright': '헤드리스 브라우저',
        'requests': 'HTTP 라이브러리',
        
        # 비디오 처리
        'yt-dlp': '비디오 다운로더',
        'opencv-python': '비디오 처리',
        
        # AI/ML
        'mediapipe': '포즈 추정',
        'numpy': '수치 연산',
        'pandas': '데이터 처리',
        
        # 데이터베이스
        'sqlalchemy': 'ORM',
        
        # 클라우드
        'boto3': 'AWS SDK',
        
        # 테스트
        'pytest': '테스트 프레임워크',
        
        # 유틸리티
        'loguru': '로깅',
        'python-dotenv': '환경 변수'
    }
    
    missing_packages = []
    for package, description in essential_packages.items():
        if package not in content:
            missing_packages.append(f"{package} ({description})")
    
    assert not missing_packages, f"필수 패키지 누락: {', '.join(missing_packages)}"


def test_package_versions_specified():
    """패키지 버전 명시 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text()
    lines = [line.strip() for line in content.split('\n') 
             if line.strip() and not line.startswith('#')]
    
    # 주요 패키지는 버전이 명시되어 있어야 함
    major_packages = ['scrapy', 'mediapipe', 'sqlalchemy', 'pytest']
    
    for package in major_packages:
        package_lines = [line for line in lines if package in line.lower()]
        assert package_lines, f"패키지 누락: {package}"
        
        # 최소한 하나의 버전 지정자가 있어야 함 (==, >=, etc.)
        has_version = any('=' in line or '>' in line for line in package_lines)
        assert has_version, f"버전이 명시되지 않음: {package}"


def test_no_duplicate_packages():
    """중복 패키지 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text()
    lines = [line.strip().split('=')[0].split('>')[0].split('<')[0] 
             for line in content.split('\n') 
             if line.strip() and not line.startswith('#')]
    
    seen = set()
    duplicates = []
    
    for package in lines:
        package_lower = package.lower().strip()
        if package_lower in seen:
            duplicates.append(package)
        seen.add(package_lower)
    
    assert not duplicates, f"중복 패키지 발견: {', '.join(duplicates)}"


def test_requirements_categories():
    """requirements.txt 카테고리 구조 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text()
    
    # 주요 카테고리가 주석으로 포함되어 있는지 확인
    expected_categories = [
        'Web Scraping',
        'Video Processing',
        'AI/ML',
        'Data Processing',
        'Database',
        'Cloud Storage',
        'Testing'
    ]
    
    for category in expected_categories:
        assert category in content, f"카테고리 주석 누락: {category}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
