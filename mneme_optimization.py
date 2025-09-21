"""
MNEME Optimization Module
Módulo de optimización de rendimiento y eficiencia de memoria
"""

import torch
import numpy as np
import time
import gc
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
import queue
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Niveles de optimización"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento"""
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 1.0
    synthesis_time: float = 0.0
    total_operations: int = 0
    failed_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_usage_mb": self.memory_usage,
            "cpu_usage_percent": self.cpu_usage,
            "gpu_usage_percent": self.gpu_usage,
            "cache_hit_rate": self.cache_hit_rate,
            "compression_ratio": self.compression_ratio,
            "synthesis_time_ms": self.synthesis_time * 1000,
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": (self.total_operations - self.failed_operations) / max(1, self.total_operations)
        }

class MemoryManager:
    """Gestor de memoria optimizado"""
    
    def __init__(self, max_memory_mb: int = 1024, 
                 gc_threshold: float = 0.8,
                 optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convertir a bytes
        self.gc_threshold = gc_threshold
        self.optimization_level = optimization_level
        self.process = psutil.Process()
        
        # Estadísticas de memoria
        self.memory_stats = {
            "peak_usage": 0.0,
            "current_usage": 0.0,
            "gc_count": 0,
            "memory_pressure_events": 0
        }
        
        # Referencias débiles para limpieza automática
        self.weak_refs: List[weakref.ref] = []
        
        # Thread para monitoreo de memoria
        self.monitor_thread = None
        self.monitoring = False
        
        if optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Iniciar monitoreo de memoria en background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _memory_monitor_loop(self):
        """Loop de monitoreo de memoria"""
        while self.monitoring:
            try:
                current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                self.memory_stats["current_usage"] = current_memory
                self.memory_stats["peak_usage"] = max(self.memory_stats["peak_usage"], current_memory)
                
                # Verificar presión de memoria
                memory_ratio = current_memory / (self.max_memory / (1024 * 1024))
                if memory_ratio > self.gc_threshold:
                    self._trigger_gc()
                    self.memory_stats["memory_pressure_events"] += 1
                
                time.sleep(1.0)  # Monitorear cada segundo
            except Exception as e:
                logger.error(f"Error en monitoreo de memoria: {e}")
    
    def _trigger_gc(self):
        """Activar recolección de basura"""
        gc.collect()
        self.memory_stats["gc_count"] += 1
        logger.debug("Garbage collection triggered")
    
    def register_object(self, obj: Any) -> weakref.ref:
        """Registrar objeto para limpieza automática"""
        ref = weakref.ref(obj, self._cleanup_callback)
        self.weak_refs.append(ref)
        return ref
    
    def _cleanup_callback(self, ref):
        """Callback de limpieza"""
        if ref in self.weak_refs:
            self.weak_refs.remove(ref)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de memoria"""
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        return {
            **self.memory_stats,
            "current_usage_mb": current_memory,
            "max_memory_mb": self.max_memory / (1024 * 1024),
            "utilization": current_memory / (self.max_memory / (1024 * 1024)),
            "weak_refs_count": len(self.weak_refs)
        }
    
    def cleanup(self):
        """Limpiar recursos"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Limpiar referencias débiles
        self.weak_refs.clear()
        gc.collect()

class PerformanceProfiler:
    """Profiler de rendimiento"""
    
    def __init__(self, enable_gpu_profiling: bool = True):
        self.enable_gpu_profiling = enable_gpu_profiling
        self.metrics = PerformanceMetrics()
        self.operation_times: List[float] = []
        self.memory_usage_history: List[float] = []
        
        # GPU profiling
        self.gpu_available = torch.cuda.is_available() if enable_gpu_profiling else False
        if self.gpu_available:
            self.gpu_memory_stats = {
                "allocated": 0,
                "cached": 0,
                "max_allocated": 0
            }
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager para perfilar operaciones"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        if self.gpu_available:
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
        
        try:
            yield
            self.metrics.total_operations += 1
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            operation_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.operation_times.append(operation_time)
            self.memory_usage_history.append(end_memory)
            
            if self.gpu_available:
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_delta = end_gpu_memory - start_gpu_memory
                
                self.gpu_memory_stats["allocated"] = end_gpu_memory
                self.gpu_memory_stats["max_allocated"] = max(
                    self.gpu_memory_stats["max_allocated"], 
                    end_gpu_memory
                )
            
            logger.debug(f"Operation {operation_name}: {operation_time*1000:.2f}ms, "
                        f"Memory: {memory_delta:+.1f}MB")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de rendimiento"""
        if self.operation_times:
            avg_time = np.mean(self.operation_times)
            min_time = np.min(self.operation_times)
            max_time = np.max(self.operation_times)
        else:
            avg_time = min_time = max_time = 0.0
        
        report = {
            "metrics": self.metrics.to_dict(),
            "operation_times": {
                "average_ms": avg_time * 1000,
                "min_ms": min_time * 1000,
                "max_ms": max_time * 1000,
                "total_operations": len(self.operation_times)
            },
            "memory_history": {
                "current_mb": self.memory_usage_history[-1] if self.memory_usage_history else 0,
                "peak_mb": max(self.memory_usage_history) if self.memory_usage_history else 0,
                "average_mb": np.mean(self.memory_usage_history) if self.memory_usage_history else 0
            }
        }
        
        if self.gpu_available:
            report["gpu_memory"] = self.gpu_memory_stats
        
        return report

class CacheOptimizer:
    """Optimizador de cache avanzado"""
    
    def __init__(self, cache_size: int, 
                 prefetch_size: int = 3,
                 eviction_policy: str = "lru_adaptive"):
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.eviction_policy = eviction_policy
        
        # Estadísticas de acceso
        self.access_patterns: Dict[bytes, List[float]] = {}
        self.frequency_counts: Dict[bytes, int] = {}
        self.recency_timestamps: Dict[bytes, float] = {}
        
        # Predicción de acceso
        self.access_predictor = AccessPredictor()
        
    def record_access(self, key: bytes, value_size: int):
        """Registrar acceso al cache"""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        self.frequency_counts[key] = self.frequency_counts.get(key, 0) + 1
        self.recency_timestamps[key] = current_time
        
        # Actualizar predictor
        self.access_predictor.update_pattern(key, current_time)
    
    def predict_next_accesses(self, current_key: bytes) -> List[bytes]:
        """Predecir siguientes accesos"""
        return self.access_predictor.predict(current_key, self.prefetch_size)
    
    def get_eviction_candidates(self, num_candidates: int = 1) -> List[bytes]:
        """Obtener candidatos para evicción"""
        if self.eviction_policy == "lru_adaptive":
            return self._lru_adaptive_eviction(num_candidates)
        elif self.eviction_policy == "lfu":
            return self._lfu_eviction(num_candidates)
        else:
            return self._lru_eviction(num_candidates)
    
    def _lru_eviction(self, num_candidates: int) -> List[bytes]:
        """Evicción LRU simple"""
        sorted_keys = sorted(self.recency_timestamps.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_keys[:num_candidates]]
    
    def _lfu_eviction(self, num_candidates: int) -> List[bytes]:
        """Evicción LFU (Least Frequently Used)"""
        sorted_keys = sorted(self.frequency_counts.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_keys[:num_candidates]]
    
    def _lru_adaptive_eviction(self, num_candidates: int) -> List[bytes]:
        """Evicción LRU adaptativa considerando frecuencia"""
        # Combinar recencia y frecuencia
        scores = {}
        current_time = time.time()
        
        for key in self.recency_timestamps:
            recency = current_time - self.recency_timestamps[key]
            frequency = self.frequency_counts.get(key, 0)
            
            # Score más alto = menos probable de evictar
            score = frequency / (1 + recency)
            scores[key] = score
        
        sorted_keys = sorted(scores.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_keys[:num_candidates]]

class AccessPredictor:
    """Predictor de patrones de acceso"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.access_history: List[Tuple[bytes, float]] = []
        self.transition_matrix: Dict[bytes, Dict[bytes, int]] = {}
        
    def update_pattern(self, key: bytes, timestamp: float):
        """Actualizar patrón de acceso"""
        self.access_history.append((key, timestamp))
        
        # Mantener tamaño del historial
        if len(self.access_history) > self.history_size:
            self.access_history.pop(0)
        
        # Actualizar matriz de transiciones
        if len(self.access_history) >= 2:
            prev_key = self.access_history[-2][0]
            if prev_key != key:
                if prev_key not in self.transition_matrix:
                    self.transition_matrix[prev_key] = {}
                self.transition_matrix[prev_key][key] = self.transition_matrix[prev_key].get(key, 0) + 1
    
    def predict(self, current_key: bytes, num_predictions: int = 3) -> List[bytes]:
        """Predecir siguientes accesos"""
        if current_key not in self.transition_matrix:
            return []
        
        transitions = self.transition_matrix[current_key]
        if not transitions:
            return []
        
        # Ordenar por frecuencia
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        return [key for key, _ in sorted_transitions[:num_predictions]]

class ParallelProcessor:
    """Procesador paralelo para operaciones MNEME"""
    
    def __init__(self, max_workers: int = None, 
                 use_processes: bool = False):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def parallel_synthesis(self, descriptors: List, synthesis_func: Callable) -> List:
        """Síntesis paralela de múltiples descriptores"""
        futures = []
        
        for desc in descriptors:
            future = self.executor.submit(synthesis_func, desc)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30s timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel synthesis failed: {e}")
                results.append(None)
        
        return results
    
    def parallel_compression(self, tensors: List[torch.Tensor], 
                           compression_func: Callable) -> List:
        """Compresión paralela de múltiples tensores"""
        futures = []
        
        for tensor in tensors:
            future = self.executor.submit(compression_func, tensor)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60s timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel compression failed: {e}")
                results.append(None)
        
        return results
    
    def cleanup(self):
        """Limpiar recursos"""
        self.executor.shutdown(wait=True)

class MNEMEOptimizer:
    """Optimizador principal de MNEME"""
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                 enable_profiling: bool = True,
                 enable_parallel_processing: bool = True):
        
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self.enable_parallel_processing = enable_parallel_processing
        
        # Inicializar componentes
        self.memory_manager = MemoryManager(max_memory_mb, optimization_level=optimization_level)
        self.cache_optimizer = CacheOptimizer(max_memory_mb * 1024 * 1024)
        
        if enable_profiling:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None
        
        if enable_parallel_processing:
            self.parallel_processor = ParallelProcessor()
        else:
            self.parallel_processor = None
        
        logger.info(f"MNEME Optimizer initialized with level: {optimization_level.name}")
    
    def optimize_tensor_operations(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimizar operaciones con tensores"""
        if self.profiler:
            with self.profiler.profile_operation("tensor_optimization"):
                return self._optimize_tensors(tensors)
        else:
            return self._optimize_tensors(tensors)
    
    def _optimize_tensors(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimizar tensores individualmente"""
        optimized = []
        
        for tensor in tensors:
            # Aplicar optimizaciones según el nivel
            if self.optimization_level == OptimizationLevel.BASIC:
                optimized_tensor = self._basic_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
                optimized_tensor = self._aggressive_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.MAXIMUM:
                optimized_tensor = self._maximum_optimization(tensor)
            else:
                optimized_tensor = tensor
            
            optimized.append(optimized_tensor)
        
        return optimized
    
    def _basic_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización básica"""
        # Contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def _aggressive_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización agresiva"""
        # Optimización básica
        tensor = self._basic_optimization(tensor)
        
        # Pin memory si está disponible
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        
        return tensor
    
    def _maximum_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización máxima"""
        # Optimización agresiva
        tensor = self._aggressive_optimization(tensor)
        
        # Compresión en memoria si es grande
        if tensor.numel() > 1000000:  # 1M elementos
            # Aplicar compresión temporal
            tensor = self._apply_memory_compression(tensor)
        
        return tensor
    
    def _apply_memory_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Aplicar compresión temporal en memoria"""
        # Para implementación real, usar algoritmos de compresión en memoria
        # Por ahora, solo retornar el tensor optimizado
        return tensor
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Obtener reporte de optimización"""
        report = {
            "optimization_level": self.optimization_level.value,
            "memory_stats": self.memory_manager.get_memory_stats(),
            "cache_stats": {
                "access_patterns": len(self.cache_optimizer.access_patterns),
                "frequency_counts": len(self.cache_optimizer.frequency_counts)
            }
        }
        
        if self.profiler:
            report["performance"] = self.profiler.get_performance_report()
        
        return report
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        self.memory_manager.cleanup()
        
        if self.parallel_processor:
            self.parallel_processor.cleanup()
        
        logger.info("MNEME Optimizer cleanup completed")
