"""
로깅 설정 및 유틸리티
"""

import sys
from pathlib import Path
from loguru import logger
from config import config


def setup_logger(module_name: str = "p-ade"):
    """
    로거 설정
    
    Args:
        module_name: 모듈 이름 (로그 파일명에 사용)
    """
    # 기본 핸들러 제거
    logger.remove()
    
    # 콘솔 출력 (INFO 이상)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if not config.DEBUG else "DEBUG",
        colorize=True
    )
    
    # 파일 출력 (DEBUG 이상)
    log_file = config.LOGS_DIR / f"{module_name}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # 에러 로그 (ERROR 이상)
    error_log_file = config.LOGS_DIR / f"{module_name}_error.log"
    logger.add(
        error_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        compression="zip"
    )
    
    return logger


# 기본 로거 초기화
setup_logger()
