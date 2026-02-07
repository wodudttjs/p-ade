"""
Prometheus Metrics Exporter

/metrics HTTP 엔드포인트 제공
"""

import os
from typing import Optional

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class MetricsExporter:
    """
    Prometheus 메트릭 HTTP Exporter
    
    Task 6.1.2: Metrics endpoint
    """
    
    def __init__(
        self,
        port: int = 9090,
        host: str = "0.0.0.0",
    ):
        self.port = port
        self.host = host
        self._server = None
        
    def start(self):
        """메트릭 서버 시작"""
        try:
            from prometheus_client import start_http_server
            
            start_http_server(self.port, addr=self.host)
            logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")
            
        except ImportError:
            logger.warning("prometheus_client not installed. Metrics endpoint not available.")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            
    def get_metrics_text(self) -> str:
        """메트릭 텍스트 포맷으로 반환 (테스트용)"""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest().decode("utf-8")
        except ImportError:
            return ""


def create_fastapi_metrics_endpoint():
    """
    FastAPI용 메트릭 엔드포인트 생성
    
    사용법:
        app = FastAPI()
        app.include_router(create_fastapi_metrics_endpoint())
    """
    try:
        from fastapi import APIRouter, Response
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        router = APIRouter()
        
        @router.get("/metrics")
        async def metrics():
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
            
        return router
        
    except ImportError:
        logger.warning("fastapi not installed")
        return None


def create_flask_metrics_endpoint(app):
    """
    Flask용 메트릭 엔드포인트 생성
    
    사용법:
        app = Flask(__name__)
        create_flask_metrics_endpoint(app)
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        @app.route("/metrics")
        def metrics():
            from flask import Response
            return Response(
                generate_latest(),
                mimetype=CONTENT_TYPE_LATEST,
            )
            
    except ImportError:
        logger.warning("prometheus_client not installed")


class PushGatewayExporter:
    """
    Prometheus Pushgateway Exporter
    
    Celery 워커 등 단기 작업용
    """
    
    def __init__(
        self,
        gateway_url: str = "localhost:9091",
        job_name: str = "p-ade-worker",
    ):
        self.gateway_url = gateway_url
        self.job_name = job_name
        
    def push(self, grouping_key: Optional[dict] = None):
        """메트릭을 Pushgateway로 전송"""
        try:
            from prometheus_client import push_to_gateway, REGISTRY
            
            push_to_gateway(
                self.gateway_url,
                job=self.job_name,
                registry=REGISTRY,
                grouping_key=grouping_key,
            )
            
        except ImportError:
            logger.warning("prometheus_client not installed")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
            
    def delete(self, grouping_key: Optional[dict] = None):
        """Pushgateway에서 메트릭 삭제"""
        try:
            from prometheus_client import delete_from_gateway
            
            delete_from_gateway(
                self.gateway_url,
                job=self.job_name,
                grouping_key=grouping_key,
            )
            
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Failed to delete metrics: {e}")
