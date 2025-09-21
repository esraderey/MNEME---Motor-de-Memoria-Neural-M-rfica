"""
MNEME Core: Motor de Memoria Neural Mórfica
Sistema avanzado de memoria computacional con síntesis determinista, verificación criptográfica y optimizaciones de hardware
"""

import hashlib
import struct
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
from collections import deque, OrderedDict
from threading import Lock, RLock
import time
import pickle
import lz4.frame
import tensorly as tl
from tensorly.decomposition import parafac, tucker, matrix_product_state
import xxhash
import io
import mmap
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import secrets
import hmac
import json
from pathlib import Path
import logging
from contextlib import contextmanager
import gc
import psutil
import warnings

# Configure TensorLy backend
tl.set_backend('pytorch')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecompType(Enum):
    """Tipos de descomposición soportados"""
    TT = "tt"           # Tensor-Train 
    CP = "cp"           # CANDECOMP/PARAFAC
    TUCKER = "tucker"   # Tucker decomposition
    SVD = "svd"         # SVD decomposition
    RAW = "raw"         # Sin compresión
    SPARSE = "sparse"   # Representación dispersa
    QUANTIZED = "quantized"  # Cuantización
    ADAPTIVE = "adaptive"   # Selección automática

class CompressionLevel(Enum):
    """Niveles de compresión"""
    ULTRA_FAST = 1
    FAST = 2
    BALANCED = 3
    HIGH = 4
    MAXIMUM = 5

@dataclass(frozen=True)
class ZDescriptor:
    """Descriptor inmutable que identifica contenido sintetizable"""
    kind: str
    decomp_type: DecompType
    shape: Tuple[int, ...]
    ranks: Optional[Tuple[int, ...]]
    seed: bytes
    version: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    delta_chain: Optional[bytes] = None
    merkle_root: Optional[bytes] = None  # Raíz del árbol Merkle
    checksum: Optional[bytes] = None     # Checksum criptográfico
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    
    def __hash__(self):
        return hash((self.kind, self.decomp_type, self.shape, self.ranks, 
                    self.seed, self.version, self.merkle_root))
    
    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor"""
        if self.checksum:
            computed = self._compute_checksum()
            return hmac.compare_digest(computed, self.checksum)
        return True
    
    def _compute_checksum(self) -> bytes:
        """Calcular checksum criptográfico"""
        h = hashlib.sha256()
        h.update(self.seed)
        if self.delta_chain:
            h.update(self.delta_chain)
        if self.merkle_root:
            h.update(self.merkle_root)
        return h.digest()

class MerkleTree:
    """Árbol Merkle para verificación de integridad"""
    
    def __init__(self, data_chunks: List[bytes]):
        self.leaves = [hashlib.sha256(chunk).digest() for chunk in data_chunks]
        self.tree = self._build_tree()
        self.root = self.tree[0] if self.tree else b''
    
    def _build_tree(self) -> List[bytes]:
        """Construir árbol Merkle"""
        if not self.leaves:
            return []
        
        current_level = self.leaves.copy()
        tree = [current_level]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = left + right
                next_level.append(hashlib.sha256(combined).digest())
            current_level = next_level
            tree.append(current_level)
        
        return tree
    
    def get_proof(self, index: int) -> List[bytes]:
        """Obtener prueba de Merkle para un índice"""
        if not self.tree or index >= len(self.leaves):
            return []
        
        proof = []
        current_index = index
        
        for level in self.tree[:-1]:  # Excluir la raíz
            if current_index % 2 == 0:  # Nodo izquierdo
                sibling_index = current_index + 1
            else:  # Nodo derecho
                sibling_index = current_index - 1
            
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            
            current_index //= 2
        
        return proof
    
    def verify_proof(self, leaf: bytes, proof: List[bytes], index: int) -> bool:
        """Verificar prueba de Merkle"""
        current_hash = hashlib.sha256(leaf).digest()
        
        for sibling in proof:
            if index % 2 == 0:
                combined = current_hash + sibling
            else:
                combined = sibling + current_hash
            current_hash = hashlib.sha256(combined).digest()
            index //= 2
        
        return current_hash == self.root

class ZAddr:
    """Zero-address: direccionamiento basado en contenido"""
    
    @staticmethod
    def compute(desc: ZDescriptor) -> bytes:
        """Calcular dirección determinista desde descriptor"""
        h = xxhash.xxh3_128()
        
        # Serializar descriptor de forma determinista
        h.update(desc.kind.encode())
        h.update(desc.decomp_type.value.encode())
        h.update(struct.pack(f'{len(desc.shape)}I', *desc.shape))
        
        if desc.ranks:
            h.update(struct.pack(f'{len(desc.ranks)}I', *desc.ranks))
        
        h.update(desc.seed)
        h.update(struct.pack('I', desc.version))
        h.update(desc.compression_level.value.to_bytes(1, 'little'))
        
        # Incluir campos meta en orden para determinismo
        for k, v in sorted(desc.meta.items()):
            h.update(k.encode())
            h.update(str(v).encode())
            
        if desc.delta_chain:
            h.update(desc.delta_chain)
        
        if desc.merkle_root:
            h.update(desc.merkle_root)
            
        return h.digest()

class AdvancedCompressor:
    """Compresor avanzado con múltiples algoritmos"""
    
    @staticmethod
    def compress(data: bytes, level: CompressionLevel = CompressionLevel.BALANCED) -> bytes:
        """Compresión con nivel configurable"""
        if level == CompressionLevel.ULTRA_FAST:
            return lz4.frame.compress(data, compression_level=1, content_checksum=True)
        elif level == CompressionLevel.FAST:
            return lz4.frame.compress(data, compression_level=3, content_checksum=True)
        elif level == CompressionLevel.BALANCED:
            return lz4.frame.compress(data, compression_level=5, content_checksum=True)
        elif level == CompressionLevel.HIGH:
            return lz4.frame.compress(data, compression_level=9, content_checksum=True)
        else:  # MAXIMUM
            return lz4.frame.compress(data, compression_level=12, content_checksum=True)
    
    @staticmethod
    def decompress(data: bytes) -> bytes:
        """Descompresión con verificación"""
        return lz4.frame.decompress(data)

class Quantizer:
    """Sistema de cuantización avanzado"""
    
    @staticmethod
    def quantize(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
        """Cuantizar tensor a bits especificados"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if min_val == max_val:
            return torch.zeros_like(tensor), 0.0, 1.0
        
        scale = (2**bits - 1) / (max_val - min_val)
        offset = min_val
        
        quantized = torch.clamp(
            torch.round((tensor - offset) * scale),
            0, 2**bits - 1
        ).to(torch.uint8)
        
        return quantized, offset, scale
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        """Descuantizar tensor"""
        return quantized.float() / scale + offset

class TensorDecomposer:
    """Descompositor de tensores avanzado con algoritmos optimizados"""
    
    @staticmethod
    def auto_select(tensor: torch.Tensor, 
                   target_ratio: float = 0.1,
                   memory_limit: Optional[int] = None) -> Tuple[DecompType, Dict[str, Any]]:
        """Selección automática inteligente de descomposición"""
        shape = tensor.shape
        numel = tensor.numel()
        
        # Verificar límite de memoria
        if memory_limit and numel * 4 > memory_limit:
            return DecompType.QUANTIZED, {"bits": 8, "target_ratio": target_ratio}
        
        # Análisis de esparsidad
        sparsity = (tensor == 0).float().mean().item()
        if sparsity > 0.95:
            return DecompType.SPARSE, {"sparsity": sparsity}
        
        # Análisis de rango para matrices 2D
        if len(shape) == 2:
            rank = min(int(min(shape) * target_ratio), min(shape) // 2)
            if rank < 5:  # Para rangos muy bajos, usar cuantización
                return DecompType.QUANTIZED, {"bits": 8, "target_ratio": target_ratio}
            return DecompType.SVD, {"rank": rank}
        
        # Análisis de dimensionalidad
        if len(shape) >= 4:  # Tensores de alta dimensión
            # Usar TT con rangos adaptativos
            tt_ranks = []
            cum_prod_left = 1
            cum_prod_right = numel
            
            for i in range(len(shape) - 1):
                cum_prod_left *= shape[i]
                cum_prod_right //= shape[i]
                rank = min(cum_prod_left, cum_prod_right)
                rank = min(rank, int(rank * target_ratio))
                tt_ranks.append(max(rank, 1))
            
            return DecompType.TT, {"ranks": tuple(tt_ranks)}
        
        # Para tensores 3D, usar CP o Tucker según el tamaño
        if len(shape) == 3:
            if numel > 1000000:  # Tensores grandes
                return DecompType.TT, {"ranks": (min(shape) // 4, min(shape) // 4)}
            else:
                rank = max(1, int(min(shape) * target_ratio))
                return DecompType.CP, {"rank": rank}
        
        # Default a cuantización para casos complejos
        return DecompType.QUANTIZED, {"bits": 8, "target_ratio": target_ratio}
    
    @staticmethod
    def decompose(tensor: torch.Tensor, 
                 decomp_type: DecompType,
                 **params) -> Dict[str, Any]:
        """Realizar descomposición de tensor con optimizaciones"""
        
        try:
            if decomp_type == DecompType.TT:
                ranks = params.get('ranks')
                if not ranks:
                    ranks = [min(10, s) for s in tensor.shape[:-1]]
                
                factors = matrix_product_state(tensor, rank=ranks, 
                                             n_iter_max=200, tol=1e-8)
                return {"factors": factors, "type": "tt", "ranks": ranks}
                
            elif decomp_type == DecompType.CP:
                rank = params.get('rank', 10)
                factors = parafac(tensor, rank=rank, init='svd', 
                               n_iter_max=300, tol=1e-8, 
                               linesearch=True)
                return {"factors": factors, "type": "cp", "rank": rank}
                
            elif decomp_type == DecompType.TUCKER:
                ranks = params.get('ranks', [min(s, 20) for s in tensor.shape])
                core, factors = tucker(tensor, rank=ranks, 
                                     n_iter_max=200, tol=1e-8)
                return {"core": core, "factors": factors, "type": "tucker", "ranks": ranks}
                
            elif decomp_type == DecompType.SVD:
                rank = params.get('rank', min(tensor.shape) // 2)
                U, S, V = torch.svd_lowrank(tensor, q=rank, niter=10)
                return {"U": U, "S": S, "V": V, "type": "svd", "rank": rank}
                
            elif decomp_type == DecompType.SPARSE:
                indices = torch.nonzero(tensor, as_tuple=False)
                values = tensor[indices.split(1, dim=1)]
                return {"indices": indices, "values": values, 
                       "shape": tensor.shape, "type": "sparse"}
            
            elif decomp_type == DecompType.QUANTIZED:
                bits = params.get('bits', 8)
                quantized, offset, scale = Quantizer.quantize(tensor, bits)
                return {"quantized": quantized, "offset": offset, "scale": scale,
                       "type": "quantized", "bits": bits}
            
            else:  # RAW
                return {"tensor": tensor, "type": "raw"}
                
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, falling back to raw")
            return {"tensor": tensor, "type": "raw"}
    
    @staticmethod
    def reconstruct(components: Dict[str, Any]) -> torch.Tensor:
        """Reconstruir tensor desde componentes descompuestos"""
        
        comp_type = components["type"]
        
        try:
            if comp_type == "tt":
                factors = components["factors"]
                return tl.tt_to_tensor(factors)
                
            elif comp_type == "cp":
                factors = components["factors"]
                return tl.cp_to_tensor(factors)
                
            elif comp_type == "tucker":
                core = components["core"]
                factors = components["factors"]
                return tl.tucker_to_tensor((core, factors))
                
            elif comp_type == "svd":
                U, S, V = components["U"], components["S"], components["V"]
                return U @ torch.diag(S) @ V.T
                
            elif comp_type == "sparse":
                shape = components["shape"]
                indices = components["indices"]
                values = components["values"]
                tensor = torch.zeros(shape, dtype=values.dtype)
                tensor[indices.split(1, dim=1)] = values
                return tensor
            
            elif comp_type == "quantized":
                quantized = components["quantized"]
                offset = components["offset"]
                scale = components["scale"]
                return Quantizer.dequantize(quantized, offset, scale)
                
            else:  # raw
                return components["tensor"]
                
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise

class ZCache:
    """Cache de alto rendimiento con LRU, prefetching y gestión inteligente de memoria"""
    
    def __init__(self, capacity_bytes: int = 1 << 30, 
                 prefetch_size: int = 3,
                 eviction_policy: str = "lru"):
        self.capacity = capacity_bytes
        self.used = 0
        self.cache = OrderedDict()
        self.lock = RLock()
        self.prefetch_size = prefetch_size
        self.eviction_policy = eviction_policy
        
        # Estadísticas avanzadas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "prefetch_hits": 0,
            "memory_pressure": 0
        }
        
        # Gestión de memoria del sistema
        self.memory_monitor = psutil.Process()
        self.memory_threshold = 0.8  # 80% de uso de RAM
        
    def get(self, key: bytes) -> Optional[torch.Tensor]:
        """Obtener tensor del cache"""
        with self.lock:
            if key in self.cache:
                # Mover al final (más recientemente usado)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key]
            self.stats["misses"] += 1
            return None
    
    def put(self, key: bytes, tensor: torch.Tensor):
        """Almacenar tensor en cache"""
        with self.lock:
            tensor_bytes = tensor.element_size() * tensor.nelement()
            
            # Verificar presión de memoria del sistema
            if self._check_memory_pressure():
                self._aggressive_eviction()
            
            # Evictar hasta tener espacio
            while self.used + tensor_bytes > self.capacity and self.cache:
                self._evict_one()
            
            if self.used + tensor_bytes <= self.capacity:
                self.cache[key] = tensor
                self.used += tensor_bytes
    
    def _check_memory_pressure(self) -> bool:
        """Verificar presión de memoria del sistema"""
        try:
            memory_percent = self.memory_monitor.memory_percent() / 100.0
            if memory_percent > self.memory_threshold:
                self.stats["memory_pressure"] += 1
                return True
        except:
            pass
        return False
    
    def _aggressive_eviction(self):
        """Evicción agresiva bajo presión de memoria"""
        # Evictar 50% del cache
        target_evictions = len(self.cache) // 2
        for _ in range(target_evictions):
            if self.cache:
                self._evict_one()
    
    def _evict_one(self):
        """Evictar un elemento según la política"""
        if not self.cache:
            return
            
        if self.eviction_policy == "lru":
            key, tensor = self.cache.popitem(last=False)
        else:  # random
            key = next(iter(self.cache))
            tensor = self.cache.pop(key)
        
        self.used -= tensor.element_size() * tensor.nelement()
        self.stats["evictions"] += 1
    
    def prefetch(self, keys: List[bytes], zgen):
        """Prefetch múltiples claves"""
        for key in keys[:self.prefetch_size]:
            if key not in self.cache:
                # En implementación real, necesitaríamos lookup table
                pass
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del cache"""
        hit_rate = self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "used_bytes": self.used,
            "capacity_bytes": self.capacity,
            "utilization": self.used / self.capacity
        }

class MarkovPrefetcher:
    """Prefetcher Markov de 2do orden con aprendizaje adaptativo"""
    
    def __init__(self, history_size: int = 2000, confidence_threshold: float = 0.3):
        self.history = deque(maxlen=history_size)
        self.transitions = {}  # (prev, curr) -> {next: count}
        self.confidence_threshold = confidence_threshold
        self.lock = Lock()
        
    def record_access(self, addr: bytes):
        """Registrar acceso para aprendizaje"""
        with self.lock:
            self.history.append(addr)
            
            if len(self.history) >= 3:
                prev = self.history[-3]
                curr = self.history[-2]
                next_addr = self.history[-1]
                
                key = (prev, curr)
                if key not in self.transitions:
                    self.transitions[key] = {}
                self.transitions[key][next_addr] = self.transitions[key].get(next_addr, 0) + 1
    
    def predict_next(self, curr: bytes, prev: Optional[bytes] = None) -> List[Tuple[bytes, float]]:
        """Predecir siguientes accesos con confianza"""
        with self.lock:
            if prev and (prev, curr) in self.transitions:
                transitions = self.transitions[(prev, curr)]
                total = sum(transitions.values())
                
                predictions = []
                for next_addr, count in transitions.items():
                    confidence = count / total
                    if confidence >= self.confidence_threshold:
                        predictions.append((next_addr, confidence))
                
                # Ordenar por confianza
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions[:5]  # Top 5 predicciones
            return []

class ZGen:
    """Motor de síntesis - el corazón de MNEME"""
    
    def __init__(self, cache_size: int = 1 << 30, 
                 compression_level: CompressionLevel = CompressionLevel.BALANCED,
                 enable_merkle: bool = True,
                 enable_checksums: bool = True):
        self.cache = ZCache(cache_size)
        self.prefetcher = MarkovPrefetcher()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.pending_synthesis = {}
        self.lock = Lock()
        self.compression_level = compression_level
        self.enable_merkle = enable_merkle
        self.enable_checksums = enable_checksums
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            "synthesis_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_ratios": []
        }
    
    def synthesize(self, desc: ZDescriptor) -> torch.Tensor:
        """Pipeline de síntesis principal con optimizaciones"""
        start_time = time.time()
        
        try:
            # Verificar integridad si está habilitado
            if self.enable_checksums and not desc.verify_integrity():
                raise ValueError("Descriptor integrity check failed")
            
            # Generar estado aleatorio determinista
            seed_int = int.from_bytes(desc.seed[:4], 'little')
            torch.manual_seed(seed_int)
            np.random.seed(seed_int)
            
            if desc.decomp_type == DecompType.RAW:
                # Para datos raw, deserializar directamente
                tensor_bytes = AdvancedCompressor.decompress(desc.seed)
                tensor = pickle.loads(tensor_bytes)
            else:
                # Deserializar componentes
                components_bytes = AdvancedCompressor.decompress(desc.seed)
                components = pickle.loads(components_bytes)
                
                # Reconstruir tensor
                tensor = TensorDecomposer.reconstruct(components)
                
                # Aplicar cadena de deltas si existe
                if desc.delta_chain:
                    deltas = pickle.loads(AdvancedCompressor.decompress(desc.delta_chain))
                    for delta_op in deltas:
                        tensor = self._apply_delta(tensor, delta_op)
            
            # Asegurar forma correcta
            if tensor.shape != desc.shape:
                tensor = tensor.reshape(desc.shape)
            
            # Actualizar estadísticas
            synthesis_time = time.time() - start_time
            self.performance_stats["synthesis_time"] += synthesis_time
            
            return tensor
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _apply_delta(self, tensor: torch.Tensor, delta_op: Dict) -> torch.Tensor:
        """Aplicar operación delta reversible"""
        op_type = delta_op["type"]
        
        if op_type == "add":
            return tensor + delta_op["value"]
        elif op_type == "mul":
            return tensor * delta_op["value"]
        elif op_type == "sparse_update":
            indices = delta_op["indices"]
            values = delta_op["values"]
            tensor = tensor.clone()
            tensor[indices] = values
            return tensor
        elif op_type == "quantized_update":
            # Actualización cuantizada
            indices = delta_op["indices"]
            quantized_values = delta_op["quantized_values"]
            offset = delta_op["offset"]
            scale = delta_op["scale"]
            values = Quantizer.dequantize(quantized_values, offset, scale)
            tensor = tensor.clone()
            tensor[indices] = values
            return tensor
        else:
            raise ValueError(f"Unknown delta op: {op_type}")
    
    def load(self, desc: ZDescriptor) -> torch.Tensor:
        """Cargar con cache y prefetching"""
        addr = ZAddr.compute(desc)
        
        # Verificar cache
        tensor = self.cache.get(addr)
        if tensor is not None:
            self.performance_stats["cache_hits"] += 1
            self.prefetcher.record_access(addr)
            return tensor
        
        self.performance_stats["cache_misses"] += 1
        
        # Verificar si la síntesis ya está en progreso
        with self.lock:
            if addr in self.pending_synthesis:
                future = self.pending_synthesis[addr]
            else:
                # Iniciar síntesis
                future = self.executor.submit(self.synthesize, desc)
                self.pending_synthesis[addr] = future
        
        # Esperar síntesis
        tensor = future.result()
        
        # Cachear resultado
        self.cache.put(addr, tensor)
        
        # Limpiar y prefetch
        with self.lock:
            self.pending_synthesis.pop(addr, None)
        
        self.prefetcher.record_access(addr)
        self._trigger_prefetch(addr)
        
        return tensor
    
    def _trigger_prefetch(self, current_addr: bytes):
        """Activar prefetching especulativo"""
        predicted = self.prefetcher.predict_next(current_addr)
        if predicted:
            # En implementación real, necesitaríamos tabla de lookup
            pass
    
    def store(self, tensor: torch.Tensor, 
             target_ratio: float = 0.1,
             decomp_type: Optional[DecompType] = None,
             memory_limit: Optional[int] = None) -> ZDescriptor:
        """Almacenar tensor creando descriptor"""
        
        # Selección automática de descomposición si no se especifica
        if decomp_type is None:
            decomp_type, params = TensorDecomposer.auto_select(tensor, target_ratio, memory_limit)
        else:
            params = {"rank": 10}
        
        # Descomponer
        components = TensorDecomposer.decompose(tensor, decomp_type, **params)
        
        # Serializar y comprimir componentes
        components_bytes = pickle.dumps(components)
        compressed = AdvancedCompressor.compress(components_bytes, self.compression_level)
        
        # Crear árbol Merkle si está habilitado
        merkle_root = None
        if self.enable_merkle:
            merkle = MerkleTree([compressed])
            merkle_root = merkle.root
        
        # Calcular checksum si está habilitado
        checksum = None
        if self.enable_checksums:
            h = hashlib.sha256()
            h.update(compressed)
            if merkle_root:
                h.update(merkle_root)
            checksum = h.digest()
        
        # Crear descriptor
        desc = ZDescriptor(
            kind="tensor",
            decomp_type=decomp_type,
            shape=tuple(tensor.shape),
            ranks=params.get("ranks") or (params.get("rank"),) if "rank" in params else None,
            seed=compressed,
            version=0,
            meta={
                "original_dtype": str(tensor.dtype),
                "compression_ratio": len(compressed) / (tensor.nelement() * tensor.element_size()),
                "decomp_params": params
            },
            merkle_root=merkle_root,
            checksum=checksum,
            compression_level=self.compression_level
        )
        
        # Registrar ratio de compresión
        ratio = desc.meta["compression_ratio"]
        self.performance_stats["compression_ratios"].append(ratio)
        
        return desc
    
    def get_performance_stats(self) -> Dict:
        """Obtener estadísticas de rendimiento"""
        cache_stats = self.cache.get_stats()
        return {
            **self.performance_stats,
            "cache": cache_stats,
            "avg_compression_ratio": np.mean(self.performance_stats["compression_ratios"]) if self.performance_stats["compression_ratios"] else 0.0
        }

class ZSpace:
    """Interfaz principal del runtime MNEME"""
    
    def __init__(self, cache_size: int = 1 << 30, 
                 compression_level: CompressionLevel = CompressionLevel.BALANCED,
                 enable_merkle: bool = True,
                 enable_checksums: bool = True):
        self.gen = ZGen(cache_size, compression_level, enable_merkle, enable_checksums)
        self.descriptor_table = {}  # name -> descriptor
        self.version_graph = {}  # track version lineage
        self.lock = Lock()
        
        # Configuración de logging
        logger.info(f"MNEME initialized with cache_size={cache_size//1024//1024}MB, "
                   f"compression_level={compression_level.name}, "
                   f"merkle={enable_merkle}, checksums={enable_checksums}")
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor y obtener su descriptor"""
        with self.lock:
            desc = self.gen.store(tensor, **kwargs)
            self.descriptor_table[name] = desc
            logger.info(f"Registered '{name}' with compression ratio {desc.meta.get('compression_ratio', 0):.3f}")
            return desc
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor por nombre"""
        with self.lock:
            if name not in self.descriptor_table:
                raise KeyError(f"Unknown tensor: {name}")
            return self.gen.load(self.descriptor_table[name])
    
    def load_desc(self, desc: ZDescriptor) -> torch.Tensor:
        """Cargar tensor desde descriptor"""
        return self.gen.load(desc)
    
    def update(self, name: str, delta_op: Dict) -> ZDescriptor:
        """Actualizar tensor con operación delta"""
        with self.lock:
            if name not in self.descriptor_table:
                raise KeyError(f"Unknown tensor: {name}")
                
            old_desc = self.descriptor_table[name]
            
            # Crear nueva cadena de deltas
            if old_desc.delta_chain:
                deltas = pickle.loads(AdvancedCompressor.decompress(old_desc.delta_chain))
            else:
                deltas = []
            
            deltas.append(delta_op)
            compressed_deltas = AdvancedCompressor.compress(pickle.dumps(deltas), self.gen.compression_level)
            
            # Crear nueva versión
            new_desc = ZDescriptor(
                kind=old_desc.kind,
                decomp_type=old_desc.decomp_type,
                shape=old_desc.shape,
                ranks=old_desc.ranks,
                seed=old_desc.seed,
                version=old_desc.version + 1,
                meta=old_desc.meta,
                delta_chain=compressed_deltas,
                merkle_root=old_desc.merkle_root,
                checksum=old_desc.checksum,
                compression_level=old_desc.compression_level
            )
            
            self.descriptor_table[name] = new_desc
            
            # Track lineage
            new_addr = ZAddr.compute(new_desc)
            old_addr = ZAddr.compute(old_desc)
            self.version_graph[new_addr] = old_addr
            
            logger.info(f"Updated '{name}' to version {new_desc.version}")
            return new_desc
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del runtime"""
        performance_stats = self.gen.get_performance_stats()
        return {
            "descriptors": len(self.descriptor_table),
            "versions": len(self.version_graph),
            "performance": performance_stats
        }
    
    def cleanup(self):
        """Limpiar recursos"""
        self.gen.executor.shutdown(wait=True)
        gc.collect()
        logger.info("MNEME cleanup completed")

# Alias para compatibilidad
Mneme = ZSpace