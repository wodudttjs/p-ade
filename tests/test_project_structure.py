"""
프로젝트 구조 테스트

프로젝트 디렉토리 구조가 올바르게 생성되었는지 확인합니다.
"""

import pytest
from pathlib import Path


def test_project_root_exists():
    """프로젝트 루트 디렉토리 존재 확인"""
    project_root = Path(__file__).parent.parent
    assert project_root.exists()
    assert project_root.is_dir()


def test_required_directories_exist():
    """필수 디렉토리 존재 확인"""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "spiders",
        "ingestion",
        "extraction",
        "transformation",
        "storage",
        "monitoring",
        "models",
        "core",
        "config",
        "data",
        "logs",
        "tests"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"디렉토리가 존재하지 않습니다: {dir_name}"
        assert dir_path.is_dir(), f"파일이 디렉토리가 아닙니다: {dir_name}"


def test_data_subdirectories_exist():
    """data 하위 디렉토리 존재 확인"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    subdirs = ["raw", "processed", "episodes"]
    
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        assert subdir_path.exists(), f"data/{subdir} 디렉토리가 존재하지 않습니다"
        assert subdir_path.is_dir()


def test_required_files_exist():
    """필수 파일 존재 확인"""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "__init__.py"
    ]
    
    for file_name in required_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"파일이 존재하지 않습니다: {file_name}"
        assert file_path.is_file(), f"디렉토리가 파일이 아닙니다: {file_name}"


def test_models_module_exists():
    """models 모듈 존재 확인"""
    project_root = Path(__file__).parent.parent
    models_init = project_root / "models" / "__init__.py"
    models_db = project_root / "models" / "database.py"
    
    assert models_init.exists()
    assert models_db.exists()


def test_config_module_exists():
    """config 모듈 존재 확인"""
    project_root = Path(__file__).parent.parent
    config_init = project_root / "config" / "__init__.py"
    config_settings = project_root / "config" / "settings.py"
    
    assert config_init.exists()
    assert config_settings.exists()


def test_core_module_exists():
    """core 모듈 존재 확인"""
    project_root = Path(__file__).parent.parent
    core_init = project_root / "core" / "__init__.py"
    core_db = project_root / "core" / "database.py"
    core_logging = project_root / "core" / "logging_config.py"
    
    assert core_init.exists()
    assert core_db.exists()
    assert core_logging.exists()


def test_requirements_file_content():
    """requirements.txt 내용 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text()
    
    # 주요 패키지가 포함되어 있는지 확인
    essential_packages = [
        "scrapy",
        "yt-dlp",
        "mediapipe",
        "numpy",
        "pandas",
        "sqlalchemy",
        "boto3",
        "pytest"
    ]
    
    for package in essential_packages:
        assert package in content.lower(), f"필수 패키지가 누락되었습니다: {package}"


def test_gitignore_file_content():
    """.gitignore 파일 내용 확인"""
    project_root = Path(__file__).parent.parent
    gitignore_file = project_root / ".gitignore"
    
    content = gitignore_file.read_text()
    
    # 주요 패턴이 포함되어 있는지 확인
    essential_patterns = [
        "__pycache__",
        ".env",
        "*.log",
        "*.mp4",
        "venv"
    ]
    
    for pattern in essential_patterns:
        assert pattern in content, f"필수 패턴이 누락되었습니다: {pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
