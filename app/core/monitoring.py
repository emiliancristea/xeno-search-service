import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import get_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error: Optional[str] = None, **metadata):
        """Mark the operation as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
        self.metadata.update(metadata)


class PrometheusMetrics:
    """Prometheus metrics collection"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'xeno_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'xeno_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Search metrics
        self.search_requests_total = Counter(
            'xeno_search_requests_total',
            'Total search requests',
            ['search_type'],
            registry=self.registry
        )
        
        self.search_duration = Histogram(
            'xeno_search_duration_seconds',
            'Search request duration',
            ['search_type'],
            registry=self.registry
        )
        
        self.search_results_count = Histogram(
            'xeno_search_results_count',
            'Number of search results returned',
            ['search_type'],
            registry=self.registry
        )
        
        # Scraping metrics
        self.scraping_requests_total = Counter(
            'xeno_scraping_requests_total',
            'Total scraping requests',
            ['status'],
            registry=self.registry
        )
        
        self.scraping_duration = Histogram(
            'xeno_scraping_duration_seconds',
            'Scraping request duration',
            registry=self.registry
        )
        
        # NLP metrics
        self.nlp_operations_total = Counter(
            'xeno_nlp_operations_total',
            'Total NLP operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.nlp_operation_duration = Histogram(
            'xeno_nlp_operation_duration_seconds',
            'NLP operation duration',
            ['operation'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'xeno_cache_operations_total',
            'Total cache operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'xeno_cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'xeno_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'xeno_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'xeno_errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_health_check: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
        
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.checks[name] = check_func
        
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else {"success": result}
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }
                overall_healthy = False
                logger.error("Health check failed", check=name, error=str(e))
        
        health_report = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
        
        self.last_health_check = datetime.utcnow()
        self.health_status = health_report
        
        return health_report


class EnhancedMonitor:
    """Enhanced monitoring system with metrics, logging, and health checks"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self.performance_log: List[PerformanceMetrics] = []
        self.max_performance_log_size = 10000
        
        # Setup logging
        self._setup_logging()
        
        # Register default health checks
        self._register_default_health_checks()
        
    def _setup_logging(self):
        """Setup structured logging configuration"""
        log_level = self.config.logging_level.upper()
        
        # Configure root logger
        import logging
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='%(message)s'
        )
        
        # File logging if configured
        if self.config.logging_file:
            file_path = Path(self.config.logging_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, log_level))
            logging.getLogger().addHandler(file_handler)
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        self.health_checker.register_check("memory", self._check_memory)
        self.health_checker.register_check("cache", self._check_cache)
        
    def _check_memory(self) -> bool:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.metrics.memory_usage.set(memory.used)
            return memory.percent < 90  # Consider healthy if < 90% usage
        except ImportError:
            return True  # Skip if psutil not available
        except Exception:
            return False
    
    async def _check_cache(self) -> bool:
        """Check cache connectivity"""
        try:
            from app.core.cache import cache_manager
            if cache_manager.backend:
                # Try a simple cache operation
                test_key = "health_check_test"
                await cache_manager.backend.set(test_key, "test", 60)
                result = await cache_manager.backend.get(test_key)
                await cache_manager.backend.delete(test_key)
                return result == "test"
            return True  # Cache disabled is considered healthy
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return False
    
    @asynccontextmanager
    async def track_operation(self, operation: str, **metadata):
        """Context manager to track operation performance"""
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        
        try:
            yield metrics
            metrics.complete(success=True)
            
        except Exception as e:
            metrics.complete(success=False, error=str(e))
            self.record_error(operation, str(e), traceback.format_exc())
            raise
            
        finally:
            self._log_performance_metrics(metrics)
    
    def _log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        # Add to performance log
        self.performance_log.append(metrics)
        
        # Trim log if too large
        if len(self.performance_log) > self.max_performance_log_size:
            self.performance_log = self.performance_log[-self.max_performance_log_size//2:]
        
        # Log to structured logging
        logger.info(
            "Operation completed",
            operation=metrics.operation,
            duration=metrics.duration,
            success=metrics.success,
            error=metrics.error,
            **metrics.metadata
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.metrics.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_search_request(self, search_type: str, duration: float, results_count: int):
        """Record search request metrics"""
        self.metrics.search_requests_total.labels(search_type=search_type).inc()
        self.metrics.search_duration.labels(search_type=search_type).observe(duration)
        self.metrics.search_results_count.labels(search_type=search_type).observe(results_count)
    
    def record_scraping_request(self, success: bool, duration: float):
        """Record scraping request metrics"""
        status = "success" if success else "failure"
        self.metrics.scraping_requests_total.labels(status=status).inc()
        self.metrics.scraping_duration.observe(duration)
    
    def record_nlp_operation(self, operation: str, success: bool, duration: float):
        """Record NLP operation metrics"""
        status = "success" if success else "failure"
        self.metrics.nlp_operations_total.labels(operation=operation, status=status).inc()
        self.metrics.nlp_operation_duration.labels(operation=operation).observe(duration)
    
    def record_cache_operation(self, operation: str, success: bool):
        """Record cache operation metrics"""
        status = "success" if success else "failure"
        self.metrics.cache_operations_total.labels(operation=operation, status=status).inc()
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate metric"""
        self.metrics.cache_hit_rate.set(hit_rate)
    
    def record_error(self, component: str, error: str, traceback_str: Optional[str] = None):
        """Record error metrics and logging"""
        error_type = type(error).__name__ if isinstance(error, Exception) else "Unknown"
        
        self.metrics.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
        
        logger.error(
            "Error recorded",
            component=component,
            error_type=error_type,
            error=str(error),
            traceback=traceback_str
        )
    
    def set_active_connections(self, count: int):
        """Set active connections count"""
        self.metrics.active_connections.set(count)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return await self.health_checker.run_health_checks()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.metrics.registry)
    
    def get_performance_summary(self, operation: Optional[str] = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        relevant_metrics = [
            m for m in self.performance_log 
            if m.start_time >= cutoff_time and (not operation or m.operation == operation)
        ]
        
        if not relevant_metrics:
            return {"message": "No metrics found for the specified criteria"}
        
        successful_metrics = [m for m in relevant_metrics if m.success and m.duration]
        failed_metrics = [m for m in relevant_metrics if not m.success]
        
        durations = [m.duration for m in successful_metrics]
        
        summary = {
            "total_operations": len(relevant_metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(relevant_metrics) if relevant_metrics else 0,
            "performance": {}
        }
        
        if durations:
            durations.sort()
            n = len(durations)
            summary["performance"] = {
                "avg_duration": sum(durations) / n,
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p50_duration": durations[n // 2],
                "p95_duration": durations[int(n * 0.95)] if n > 20 else durations[-1],
                "p99_duration": durations[int(n * 0.99)] if n > 100 else durations[-1],
            }
        
        if failed_metrics:
            error_counts = {}
            for m in failed_metrics:
                error_type = m.error or "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            summary["errors"] = error_counts
        
        return summary


# Global monitor instance
monitor = EnhancedMonitor()


def get_monitor() -> EnhancedMonitor:
    """Get global monitor instance"""
    return monitor


# Decorator for automatic operation tracking
def track_operation(operation_name: str, **metadata):
    """Decorator to automatically track operation performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with monitor.track_operation(operation_name, **metadata) as metrics:
                    result = await func(*args, **kwargs)
                    metrics.metadata.update({"result_type": type(result).__name__})
                    return result
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    perf_metrics = PerformanceMetrics(
                        operation=operation_name,
                        start_time=start_time,
                        duration=duration,
                        success=True,
                        metadata={**metadata, "result_type": type(result).__name__}
                    )
                    monitor._log_performance_metrics(perf_metrics)
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    perf_metrics = PerformanceMetrics(
                        operation=operation_name,
                        start_time=start_time,
                        duration=duration,
                        success=False,
                        error=str(e),
                        metadata=metadata
                    )
                    monitor._log_performance_metrics(perf_metrics)
                    monitor.record_error(operation_name, str(e), traceback.format_exc())
                    raise
            return sync_wrapper
    return decorator