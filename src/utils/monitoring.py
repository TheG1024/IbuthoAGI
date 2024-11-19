"""
Monitoring and logging system for IbuthoAGI.
"""
import os
import time
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import openai
import torch
import psutil
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ibutho.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    api_latency: float
    token_usage: int
    error_count: int

class MetricsCollector:
    def __init__(self, port: int = 8000):
        # Initialize Prometheus metrics
        self.api_calls = Counter(
            'ibutho_api_calls_total',
            'Total number of API calls',
            ['agent_type', 'endpoint']
        )
        
        self.response_time = Histogram(
            'ibutho_response_time_seconds',
            'Response time in seconds',
            ['agent_type', 'operation']
        )
        
        self.token_usage = Counter(
            'ibutho_token_usage_total',
            'Total token usage',
            ['model', 'operation']
        )
        
        self.error_rate = Counter(
            'ibutho_errors_total',
            'Total number of errors',
            ['agent_type', 'error_type']
        )
        
        self.memory_usage = Gauge(
            'ibutho_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        
        self.gpu_usage = Gauge(
            'ibutho_gpu_usage_percent',
            'Current GPU usage percentage',
            ['device']
        )
        
        # Start Prometheus server
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # GPU usage if available
            gpu_usage = None
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            # Update Prometheus metrics
            self.memory_usage.set(psutil.virtual_memory().used)
            if gpu_usage is not None:
                self.gpu_usage.labels(device='cuda:0').set(gpu_usage * 100)
            
            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                gpu_usage=gpu_usage,
                api_latency=0.0,  # Will be updated by monitor_api_call
                token_usage=0,    # Will be updated by monitor_token_usage
                error_count=0     # Will be updated by monitor_error
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return SystemMetrics(0, 0, None, 0, 0, 0)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = MetricsCollector()
        self.start_time = time.time()
        logger.info("Performance monitoring initialized")
    
    def monitor_api_call(self, agent_type: str, endpoint: str) -> None:
        """Monitor API call performance."""
        self.metrics.api_calls.labels(
            agent_type=agent_type,
            endpoint=endpoint
        ).inc()
    
    def monitor_response_time(
        self,
        agent_type: str,
        operation: str,
        start_time: float
    ) -> None:
        """Monitor operation response time."""
        duration = time.time() - start_time
        self.metrics.response_time.labels(
            agent_type=agent_type,
            operation=operation
        ).observe(duration)
    
    def monitor_token_usage(
        self,
        model: str,
        operation: str,
        tokens: int
    ) -> None:
        """Monitor token usage."""
        self.metrics.token_usage.labels(
            model=model,
            operation=operation
        ).inc(tokens)
    
    def monitor_error(
        self,
        agent_type: str,
        error_type: str
    ) -> None:
        """Monitor errors."""
        self.metrics.error_rate.labels(
            agent_type=agent_type,
            error_type=error_type
        ).inc()
        logger.error(f"Error in {agent_type}: {error_type}")

class PerformanceOptimizer:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimization_history = []
    
    def optimize_model_parameters(
        self,
        current_metrics: SystemMetrics,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize model parameters based on performance metrics."""
        try:
            # Analyze system load
            system_load = (
                current_metrics.cpu_usage / 100 +
                current_metrics.memory_usage / 100
            ) / 2
            
            # Adjust parameters based on load
            if system_load > 0.8:  # High load
                model_config['batch_size'] = max(1, model_config.get('batch_size', 16) // 2)
                model_config['max_tokens'] = min(
                    model_config.get('max_tokens', 2000),
                    1000
                )
            elif system_load < 0.3:  # Low load
                model_config['batch_size'] = min(32, model_config.get('batch_size', 16) * 2)
                model_config['max_tokens'] = min(
                    4000,
                    model_config.get('max_tokens', 2000) * 1.5
                )
            
            # Log optimization
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'system_load': system_load,
                'adjustments': model_config
            })
            
            return model_config
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return model_config
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            history = np.array([
                [h['system_load'] for h in self.optimization_history],
                [h['adjustments'].get('batch_size', 0) for h in self.optimization_history]
            ])
            
            return {
                'mean_load': np.mean(history[0]),
                'load_std': np.std(history[0]),
                'batch_size_correlation': np.corrcoef(history)[0, 1],
                'optimization_count': len(self.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            return {}
