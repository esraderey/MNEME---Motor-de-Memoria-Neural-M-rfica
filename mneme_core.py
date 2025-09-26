"""
MNEME Core: Motor de Memoria Neural Mórfica (Refactorizado y Mejorado)
Sistema avanzado de memoria computacional con síntesis determinista, verificación criptográfica robusta, 
aceleración de hardware y optimizaciones de rendimiento.
"""

import hashlib
import struct
import numpy as np
import torch
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from collections import deque, OrderedDict
from threading import Lock, RLock
import time
import lz4.frame
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
import xxhash
import io
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import secrets
import hmac
import json
import logging
import gc
import psutil
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar backend de TensorLy
try:
    tl.set_backend('pytorch')
except Exception as e:
    warnings.warn(f"Could not set TensorLy backend to PyTorch: {e}")

# --- Enums y Clases de Error ---

class DecompType(Enum):
    TT = "tt"
    CP = "cp"
    TUCKER = "tucker"
    SVD = "svd"
    RAW = "raw"
    SPARSE = "sparse"
    QUANTIZED = "quantized"
    ADAPTIVE = "adaptive"

class CompressionLevel(Enum):
    # Mapeo a niveles de compresión LZ4 (1-12)
    ULTRA_FAST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9
    MAXIMUM = 12

class SecurityError(Exception):
    """Error relacionado con la seguridad (e.g., fallo de verificación HMAC)"""
    pass

# --- Configuración ---

@dataclass
class MnemeConfig:
    """Configuración centralizada para el motor MNEME"""
    cache_size_bytes: int = 1 << 30  # 1 GB
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    use_gpu: bool = True
    secret_key: Optional[bytes] = None # Para firmado HMAC
    enable_merkle: bool = False # Merkle Tree complexity might outweigh benefits for single chunks
    delta_consolidation_threshold: int = 50 # Consolidar después de 50 deltas
    memory_pressure_threshold: float = 0.85 # 85% usage
    num_workers: int = max(4, psutil.cpu_count(logical=False) or 4)

# --- Utilidades ---

def deterministic_serialize(data: Any) -> bytes:
    """Serializa metadatos de forma determinista para hashing."""
    if isinstance(data, dict):
        # JSON con claves ordenadas garantiza determinismo
        return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    elif isinstance(data, Enum):
        return str(data.value).encode('utf-8')
    # Fallback para otros tipos básicos
    return str(data).encode('utf-8')

# --- Core Data Structures ---

@dataclass(frozen=True, slots=True)
class ZDescriptor:
    """Descriptor inmutable que identifica contenido sintetizable"""
    kind: str
    decomp_type: DecompType
    shape: Tuple[int, ...]
    # core_data contiene componentes base comprimidos y firmados
    core_data: bytes 
    version: int = 0
    ranks: Optional[Tuple[int, ...]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # delta_chain contiene deltas acumulados comprimidos y firmados
    delta_chain: Optional[bytes] = None
    merkle_root: Optional[bytes] = None
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    # El checksum verifica la integridad de todo el descriptor (metadatos + contenido)
    checksum: bytes = field(init=False)

    def __post_init__(self):
        # Calcular checksum durante la inicialización. 
        # Usar object.__setattr__ porque la dataclass es frozen.
        object.__setattr__(self, 'checksum', self._compute_descriptor_checksum())

    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor completo."""
        computed = self._compute_descriptor_checksum()
        # Usar compare_digest para seguridad contra ataques de timing
        return secrets.compare_digest(computed, self.checksum)

    def _compute_content_hash(self) -> bytes:
        """Calcula el hash SHA256 del contenido real (core_data + deltas + merkle)."""
        h = hashlib.sha256()
        h.update(self.core_data)
        if self.delta_chain:
            h.update(self.delta_chain)
        if self.merkle_root:
            h.update(self.merkle_root)
        return h.digest()

    def _compute_descriptor_checksum(self) -> bytes:
        """
        [MEJORA] Calcula el checksum criptográfico (SHA256) del descriptor completo.
        """
        h = hashlib.sha256()
        
        # 1. Incluir metadatos estructurales de forma determinista
        h.update(deterministic_serialize(self.kind))
        h.update(deterministic_serialize(self.decomp_type))
        # Usar 'Q' (unsigned long long) para shapes potencialmente grandes
        h.update(struct.pack(f'!{len(self.shape)}Q', *self.shape))
        
        if self.ranks:
            h.update(struct.pack(f'!{len(self.ranks)}I', *self.ranks))
        
        h.update(struct.pack('!Q', self.version))
        h.update(deterministic_serialize(self.compression_level))
        h.update(deterministic_serialize(self.meta))

        # 2. Incluir el hash del contenido
        h.update(self._compute_content_hash())
        
        return h.digest()

class ZAddr:
    """Zero-address: direccionamiento basado en contenido"""
    
    @staticmethod
    def compute(desc: ZDescriptor) -> bytes:
        """
        Calcular dirección determinista. Usamos el checksum del descriptor, 
        ya que representa de forma única y segura el estado completo.
        Usamos XXH3-128 sobre el checksum SHA256 para una dirección rápida (128 bits).
        """
        return xxhash.xxh3_128(desc.checksum).digest()

class MerkleTree:
    """Árbol Merkle para verificación de integridad (Implementación básica)"""
    # (La implementación original era correcta. Se mantiene simplificada aquí)
    @staticmethod
    def compute_root(data_chunks: List[bytes]) -> bytes:
        if not data_chunks:
            return b''
        # En un sistema real, esto construiría el árbol. Aquí simplificamos el concepto.
        h = hashlib.sha256()
        for chunk in data_chunks:
             h.update(hashlib.sha256(chunk).digest())
        return h.digest()

class AdvancedCompressor:
    @staticmethod
    def compress(data: bytes, level: CompressionLevel) -> bytes:
        try:
            return lz4.frame.compress(data, compression_level=level.value, content_checksum=True)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    @staticmethod
    def decompress(data: bytes) -> bytes:
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            logger.error(f"Decompression failed (data corruption likely): {e}")
            raise

# --- Serialización Segura y Firmado ---

class Serializer:
    """
    [MEJORA] Maneja la serialización segura (torch.save/load) y el firmado HMAC.
    Reemplaza el uso de Pickle.
    """

    def __init__(self, secret_key: Optional[bytes]):
        self.secret_key = secret_key

    def serialize(self, data: Any) -> bytes:
        """Serializa datos (moviendo a CPU) y los firma si hay clave."""
        buffer = io.BytesIO()
        try:
            # Mover tensores a CPU antes de guardar para portabilidad
            data_cpu = self._move_to_cpu(data)
            torch.save(data_cpu, buffer)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
        
        serialized_data = buffer.getvalue()
        
        if self.secret_key:
            return self._sign_data(serialized_data)
        return serialized_data

    def deserialize(self, data: bytes, device: torch.device) -> Any:
        """Verifica firma (si existe) y deserializa datos en el dispositivo objetivo."""
        
        if self.secret_key:
            data = self._verify_and_extract_data(data)
        
        buffer = io.BytesIO(data)
        try:
            # map_location asegura que los tensores se carguen en el dispositivo correcto
            return torch.load(buffer, map_location=device)
        except Exception as e:
            logger.error(f"Deserialization failed (potential data corruption or tampering): {e}")
            raise

    def _move_to_cpu(self, data):
        """Mueve recursivamente tensores a CPU."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_cpu(v) for v in data)
        return data

    def _sign_data(self, data: bytes) -> bytes:
        """Firma datos usando HMAC-SHA256 (Formato: Firma + Datos)."""
        signature = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        return signature + data

    def _verify_and_extract_data(self, signed_data: bytes) -> bytes:
        """Verifica firma HMAC y extrae datos."""
        if len(signed_data) < 32:
            raise SecurityError("Invalid signed data format.")
            
        signature = signed_data[:32]
        data = signed_data[32:]
        
        expected_signature = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise SecurityError("Data integrity verification failed (HMAC mismatch).")
            
        return data

# --- Tensor Operations (Quantizer, Decomposer) ---

class Quantizer:
    # (Implementación similar a la original, robusta y funcional)
    @staticmethod
    def quantize(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if min_val == max_val:
            dtype = torch.uint8 if bits <= 8 else torch.int16
            return torch.zeros_like(tensor, dtype=dtype), min_val, 1.0
        
        scale = (2**bits - 1) / (max_val - min_val)
        offset = min_val
        
        quantized = torch.clamp(torch.round((tensor - offset) * scale), 0, 2**bits - 1)
        
        if bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif bits <= 16:
            quantized = quantized.to(torch.int16) # PyTorch prefiere int16 sobre uint16
        else:
            quantized = quantized.to(torch.int32)
        
        return quantized, offset, scale
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        return quantized.float() / scale + offset

class TensorDecomposer:
    """Descompositor de tensores con heurísticas mejoradas y soporte de dispositivo."""
    
    @staticmethod
    def auto_select(tensor: torch.Tensor, 
                   target_ratio: float = 0.1) -> Tuple[DecompType, Dict[str, Any]]:
        # (Heurísticas refinadas basadas en la implementación original y mejoras comunes)
        shape = tensor.shape
        numel = tensor.numel()
        
        # 1. Análisis de esparsidad
        sparsity = (tensor == 0).float().mean().item()
        if sparsity > 0.90:
            return DecompType.SPARSE, {}
        
        # 2. Análisis de Dimensionalidad
        if len(shape) == 2:
            # SVD para matrices. Heurística refinada para respetar target_ratio.
            M, N = shape
            # Ratio = K*(M+N+1) / (M*N). Resolver para K (rank).
            target_rank = int((target_ratio * M * N) / (M + N + 1))
            max_rank = min(M, N)
            rank = max(1, min(target_rank, max_rank))

            if rank < 5 and max_rank > 100:
                return DecompType.QUANTIZED, {"bits": 8}
            return DecompType.SVD, {"rank": rank}
        
        if len(shape) >= 3:
            # TT preferido para alta dimensión por su robustez
            avg_dim = np.prod(shape)**(1/len(shape))
            # Heurística simplificada para rango TT
            scaling_factor = target_ratio**(1/(len(shape)-1))
            target_rank = max(1, int(avg_dim * scaling_factor))
            tt_ranks = tuple([target_rank] * (len(shape) - 1))
            return DecompType.TT, {"ranks": tt_ranks}
        
        # Default (e.g., 1D tensors)
        return DecompType.QUANTIZED, {"bits": 8}
    
    @staticmethod
    def decompose(tensor: torch.Tensor, decomp_type: DecompType, device: torch.device, **params) -> Dict[str, Any]:
        # Mover tensor al dispositivo de cómputo
        tensor = tensor.to(device)

        try:
            if decomp_type == DecompType.TT:
                ranks = params.get('ranks')
                if not ranks:
                     # Fallback
                    avg_dim = np.prod(tensor.shape)**(1/len(tensor.shape))
                    default_rank = max(1, int(avg_dim * 0.1))
                    ranks = [default_rank] * (len(tensor.shape) - 1)
                
                factors = tensor_train(tensor, rank=ranks)
                # Extraer los factores (tensores) del objeto TensorTrain
                return {"factors": list(factors), "type": "tt", "ranks": ranks}
                
            elif decomp_type == DecompType.CP:
                rank = params.get('rank', 10)
                weights, factors = parafac(tensor, rank=rank, init='svd', n_iter_max=300, tol=1e-8, linesearch=True)
                return {"weights": weights, "factors": factors, "type": "cp", "rank": rank}
                
            elif decomp_type == DecompType.TUCKER:
                ranks = params.get('ranks', [min(s, 20) for s in tensor.shape])
                core, factors = tucker(tensor, rank=ranks, n_iter_max=200, tol=1e-8)
                return {"core": core, "factors": factors, "type": "tucker", "ranks": ranks}

            elif decomp_type == DecompType.SVD:
                rank = params.get('rank', min(tensor.shape) // 4)
                U, S, V = torch.svd_lowrank(tensor, q=rank, niter=10)
                return {"U": U, "S": S, "V": V, "type": "svd", "rank": rank}

            elif decomp_type == DecompType.SPARSE:
                sparse_tensor = tensor.to_sparse()
                return {"indices": sparse_tensor.indices(), "values": sparse_tensor.values(), "shape": tensor.shape, "type": "sparse"}

            elif decomp_type == DecompType.QUANTIZED:
                bits = params.get('bits', 8)
                quantized, offset, scale = Quantizer.quantize(tensor, bits)
                return {"quantized": quantized, "offset": offset, "scale": scale, "type": "quantized", "bits": bits}

            else:  # RAW
                return {"tensor": tensor, "type": "raw"}
                
        except Exception as e:
            logger.warning(f"Decomposition {decomp_type.value} failed on device {device}: {e}. Falling back to RAW.")
            return {"tensor": tensor, "type": "raw"}
    
    @staticmethod
    def reconstruct(components: Dict[str, Any], device: torch.device) -> torch.Tensor:
        # Los componentes ya deberían estar en el dispositivo correcto gracias al Serializer.deserialize(map_location=device)
        comp_type = components["type"]
        
        try:
            if comp_type == "tt":
                factors = components["factors"]
                return tl.tt_to_tensor(factors)
            elif comp_type == "cp":
                weights = components.get("weights")
                factors = components["factors"]
                return tl.cp_to_tensor((weights, factors))
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
                return torch.sparse_coo_tensor(indices, values, shape, device=device).to_dense()
            elif comp_type == "quantized":
                quantized = components["quantized"]
                offset = components["offset"]
                scale = components["scale"]
                return Quantizer.dequantize(quantized, offset, scale)
            else:  # raw
                return components["tensor"]
                
        except Exception as e:
            logger.error(f"Reconstruction failed for type {comp_type} on device {device}: {e}")
            raise

# --- Caching and Prefetching Systems ---

class ZCache:
    """
    [MEJORA] Cache LRU con gestión inteligente de memoria. Almacena en CPU para ahorrar VRAM.
    """
    
    def __init__(self, config: MnemeConfig):
        self.capacity = config.cache_size_bytes
        self.memory_threshold = config.memory_pressure_threshold
        self.used = 0
        self.cache = OrderedDict()
        self.lock = RLock()
        
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "memory_pressure_events": 0}
        
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def get(self, key: bytes, target_device: torch.device) -> Optional[torch.Tensor]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                # Mover al dispositivo solicitado solo en la lectura
                return self.cache[key].to(target_device)
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: bytes, tensor: torch.Tensor):
        with self.lock:
            if key in self.cache:
                # Si ya existe, no hacer nada (asumiendo inmutabilidad del contenido)
                return

            # Asegurar que el tensor esté en CPU antes de cachear
            tensor_cpu = tensor.cpu()
            tensor_bytes = self._get_tensor_size(tensor_cpu)
            
            if tensor_bytes > self.capacity:
                return

            if self._check_system_memory_pressure():
                self._aggressive_eviction()
            
            while self.used + tensor_bytes > self.capacity and self.cache:
                self._evict_one()
            
            self.cache[key] = tensor_cpu
            self.used += tensor_bytes
            self.cache.move_to_end(key)

    def _check_system_memory_pressure(self) -> bool:
        """Verificar presión de memoria del sistema (RAM)."""
        try:
            vm = psutil.virtual_memory()
            if vm.percent / 100.0 > self.memory_threshold:
                self.stats["memory_pressure_events"] += 1
                return True
        except Exception:
            pass
        return False

    def _aggressive_eviction(self):
        """Liberar 20% del cache."""
        logger.warning("System memory pressure detected. Initiating aggressive eviction.")
        target_usage = int(self.capacity * 0.8)
        while self.used > target_usage and self.cache:
            self._evict_one()
        gc.collect()
    
    def _evict_one(self):
        if not self.cache: return
        _, tensor = self.cache.popitem(last=False) # LRU
        self.used -= self._get_tensor_size(tensor)
        self.stats["evictions"] += 1

class MarkovPrefetcher:
    """Prefetcher Markov de 2do orden."""
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.transitions = {}  # (prev, curr) -> {next: count}
        self.confidence_threshold = 0.3
        self.lock = Lock()
        
    def record_access(self, addr: bytes):
        with self.lock:
            self.history.append(addr)
            if len(self.history) >= 3:
                A, B, C = self.history[-3], self.history[-2], self.history[-1]
                key = (A, B)
                if key not in self.transitions:
                    self.transitions[key] = {}
                self.transitions[key][C] = self.transitions[key].get(C, 0) + 1
    
    def predict_next(self, curr: bytes) -> List[bytes]:
        with self.lock:
            if len(self.history) < 2: return []
            
            prev = self.history[-2]
            key = (prev, curr)

            if key in self.transitions:
                transitions = self.transitions[key]
                total = sum(transitions.values())
                predictions = []
                for next_addr, count in transitions.items():
                    confidence = count / total
                    if confidence >= self.confidence_threshold:
                        predictions.append((next_addr, confidence))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                return [addr for addr, conf in predictions[:3]] # Top 3
            return []

# --- Synthesis Engine (ZGen) ---

class ZGen:
    """Motor de síntesis."""
    
    def __init__(self, config: MnemeConfig, 
                 device: torch.device,
                 descriptor_lookup_fn: Callable[[bytes], Optional[ZDescriptor]]):
        self.config = config
        self.device = device
        self.cache = ZCache(config)
        self.prefetcher = MarkovPrefetcher()
        self.serializer = Serializer(config.secret_key)
        
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.pending_synthesis: Dict[bytes, Future] = {} # addr -> Future
        self.lock = Lock()
        self.descriptor_lookup_fn = descriptor_lookup_fn
        
        self.stats = {"total_synthesis_time": 0.0, "synthesis_count": 0, "compression_ratios": []}

    def synthesize(self, desc: ZDescriptor) -> torch.Tensor:
        """Pipeline de síntesis principal."""
        start_time = time.time()
        
        try:
            # 1. Verificación de Integridad del Descriptor
            if not desc.verify_integrity():
                raise SecurityError("Descriptor integrity check failed.")
            
            # 2. Procesar Core Data (Descomprimir, Verificar Firma, Deserializar)
            core_data_bytes = AdvancedCompressor.decompress(desc.core_data)
            # Carga los componentes en el dispositivo de cómputo
            components = self.serializer.deserialize(core_data_bytes, device=self.device)
            
            # 3. Reconstrucción del Tensor Base
            tensor = TensorDecomposer.reconstruct(components, self.device)
            
            # 4. Aplicación de Cadena de Deltas
            if desc.delta_chain:
                deltas_bytes = AdvancedCompressor.decompress(desc.delta_chain)
                # Carga los deltas en el dispositivo de cómputo
                deltas = self.serializer.deserialize(deltas_bytes, device=self.device)
                
                for delta_op in deltas:
                    tensor = self._apply_delta(tensor, delta_op)
            
            # 5. Verificación Final de Forma
            if tensor.shape != desc.shape:
                try:
                    tensor = tensor.reshape(desc.shape)
                except RuntimeError:
                    raise ValueError(f"Synthesized tensor shape mismatch. Expected {desc.shape}, got {tensor.shape}.")
            
            # Actualizar estadísticas
            self.stats["total_synthesis_time"] += time.time() - start_time
            self.stats["synthesis_count"] += 1
            
            return tensor
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _apply_delta(self, tensor: torch.Tensor, delta_op: Dict) -> torch.Tensor:
        """Aplica operación delta (en el dispositivo)."""
        op_type = delta_op["type"]
        # Los valores ya están en el dispositivo correcto gracias al Serializer
        
        if op_type == "add":
            return tensor + delta_op["value"]
        elif op_type == "mul":
            return tensor * delta_op["value"]
        elif op_type == "sparse_update":
            indices = delta_op["indices"]
            values = delta_op["values"]
            tensor = tensor.clone()
            # Manejo de índices multidimensionales requiere tupla
            if indices.dim() > 1:
                tensor[tuple(indices)] = values
            else:
                tensor[indices] = values
            return tensor
        elif op_type == "quantized_update":
            indices = delta_op["indices"]
            quantized_values = delta_op["quantized_values"]
            offset = delta_op["offset"]
            scale = delta_op["scale"]
            values = Quantizer.dequantize(quantized_values, offset, scale)
            tensor = tensor.clone()
            if indices.dim() > 1:
                tensor[tuple(indices)] = values
            else:
                tensor[indices] = values
            return tensor
        else:
            raise ValueError(f"Unknown delta op: {op_type}")
    
    def load(self, desc: ZDescriptor) -> torch.Tensor:
        """Cargar con cache, manejo de concurrencia y prefetching."""
        addr = ZAddr.compute(desc)
        
        # 1. Verificar Cache
        tensor = self.cache.get(addr, self.device)
        if tensor is not None:
            self.prefetcher.record_access(addr)
            self._trigger_prefetch(addr)
            return tensor
        
        # 2. Manejo de Síntesis Concurrente (Thundering Herd Protection)
        with self.lock:
            if addr in self.pending_synthesis:
                future = self.pending_synthesis[addr]
            else:
                future = self.executor.submit(self.synthesize, desc)
                self.pending_synthesis[addr] = future
        
        # 3. Esperar Resultado y Cachear
        try:
            tensor = future.result()
            # ZCache se encarga de mover a CPU antes de almacenar
            self.cache.put(addr, tensor)
        finally:
            with self.lock:
                # Asegurar que eliminamos la future correcta
                if self.pending_synthesis.get(addr) == future:
                   self.pending_synthesis.pop(addr, None)
        
        # 4. Prefetching
        self.prefetcher.record_access(addr)
        self._trigger_prefetch(addr)
        
        return tensor
    
    def _trigger_prefetch(self, current_addr: bytes):
        """[MEJORA] Activar prefetching especulativo funcional."""
        predicted_addrs = self.prefetcher.predict_next(current_addr)
        
        for addr in predicted_addrs:
            if addr not in self.cache.cache:
                with self.lock:
                    if addr not in self.pending_synthesis:
                        # Buscar el descriptor correspondiente usando el callback
                        desc = self.descriptor_lookup_fn(addr)
                        if desc:
                            logger.debug(f"Prefetching {addr.hex()[:8]}...")
                            future = self.executor.submit(self.synthesize, desc)
                            self.pending_synthesis[addr] = future
                            # Añadir callback para cachear cuando termine
                            future.add_done_callback(lambda f: self._handle_prefetch_result(addr, f))

    def _handle_prefetch_result(self, addr: bytes, future: Future):
        """Callback para resultados de prefetch."""
        with self.lock:
             if self.pending_synthesis.get(addr) == future:
                self.pending_synthesis.pop(addr, None)
        try:
            tensor = future.result()
            self.cache.put(addr, tensor)
        except Exception:
            pass

    def store(self, tensor: torch.Tensor, 
             target_ratio: float = 0.1,
             decomp_type: Optional[DecompType] = None) -> ZDescriptor:
        """Almacenar tensor: descomponer, serializar (seguro), firmar, comprimir y crear descriptor."""
        
        # 1. Selección y Descomposición (en el dispositivo de cómputo)
        if decomp_type is None or decomp_type == DecompType.ADAPTIVE:
            decomp_type, params = TensorDecomposer.auto_select(tensor, target_ratio)
        else:
            params = {}
        
        components = TensorDecomposer.decompose(tensor, decomp_type, self.device, **params)
        
        # 2. Serialización (incluye mover a CPU y firmado) y Compresión
        components_bytes = self.serializer.serialize(components)
        compressed_core_data = AdvancedCompressor.compress(components_bytes, self.config.compression_level)
        
        # 3. Creación de Árbol Merkle (Opcional)
        merkle_root = None
        if self.config.enable_merkle:
            merkle_root = MerkleTree.compute_root([compressed_core_data])
        
        # 4. Creación del Descriptor (El checksum se calcula automáticamente en __post_init__)
        original_size = tensor.nelement() * tensor.element_size() + 1e-9
        ratio = len(compressed_core_data) / original_size

        desc = ZDescriptor(
            kind="tensor",
            decomp_type=decomp_type,
            shape=tuple(tensor.shape),
            ranks=params.get("ranks") or (params.get("rank"),) if "rank" in params else None,
            core_data=compressed_core_data,
            version=0,
            meta={"compression_ratio": ratio, "decomp_params": params, "dtype": str(tensor.dtype)},
            merkle_root=merkle_root,
            compression_level=self.config.compression_level
        )
        
        self.stats["compression_ratios"].append(ratio)
        return desc
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

# --- Main Runtime Interface (ZSpace) ---

class ZSpace:
    """Interfaz principal del runtime MNEME. Gestiona descriptores, versiones y consolidación."""
    
    def __init__(self, config: Optional[MnemeConfig] = None):
        self.config = config or MnemeConfig()
        
        # 1. Configuración de Dispositivo
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 2. Configuración de Seguridad (Clave Secreta)
        if self.config.secret_key:
            if len(self.config.secret_key) < 32: 
                raise ValueError("Secret key must be >= 32 bytes.")
        else:
            logger.warning("No secret key provided. Generating a transient secure key. Persistence requires a fixed key.")
            # Generar una clave transitoria segura si no se proporciona
            self.config = replace(self.config, secret_key=secrets.token_bytes(32))

        # 3. Tablas de mapeo
        self.name_to_desc: Dict[str, ZDescriptor] = {}
        # [MEJORA] Índice inverso para prefetching
        self.addr_to_desc: Dict[bytes, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {} # new_addr -> old_addr
        self.lock = RLock()
        
        # 4. Inicializar ZGen
        self.gen = ZGen(self.config, self.device, self._lookup_descriptor_by_addr)

        logger.info(f"MNEME ZSpace initialized on device {self.device}.")

    # [MEJORA] Implementar Context Manager para gestión de recursos
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _lookup_descriptor_by_addr(self, addr: bytes) -> Optional[ZDescriptor]:
        """Callback para ZGen."""
        with self.lock:
            return self.addr_to_desc.get(addr)

    def _register_descriptor(self, name: str, desc: ZDescriptor, old_addr: Optional[bytes] = None):
        """Helper para registrar descriptor en todas las tablas y manejar linaje."""
        addr = ZAddr.compute(desc)
        self.name_to_desc[name] = desc
        self.addr_to_desc[addr] = desc
        if old_addr:
            self.version_graph[addr] = old_addr
        return addr
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor."""
        
        desc = self.gen.store(tensor, **kwargs)
        
        with self.lock:
            old_addr = None
            if name in self.name_to_desc:
                old_addr = ZAddr.compute(self.name_to_desc[name])

            addr = self._register_descriptor(name, desc, old_addr)
            
        logger.info(f"Registered '{name}'. Type: {desc.decomp_type.value}. Ratio: {desc.meta['compression_ratio']:.3f}. Addr: {addr.hex()[:8]}")
        return desc
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor por nombre."""
        with self.lock:
            if name not in self.name_to_desc:
                raise KeyError(f"Unknown tensor: {name}")
            desc = self.name_to_desc[name]
        return self.gen.load(desc)
    
    def update(self, name: str, delta_op: Dict) -> ZDescriptor:
        """
        [MEJORA] Actualizar tensor con operación delta y manejar consolidación automática.
        """
        with self.lock:
            if name not in self.name_to_desc:
                raise KeyError(f"Unknown tensor: {name}")
                
            old_desc = self.name_to_desc[name]
            
            # 1. Cargar y Deserializar Cadena Delta Existente (Seguro)
            if old_desc.delta_chain:
                try:
                    deltas_bytes = AdvancedCompressor.decompress(old_desc.delta_chain)
                    # Deserializar a CPU para manipulación de la estructura
                    deltas = self.gen.serializer.deserialize(deltas_bytes, device=torch.device('cpu'))
                except Exception as e:
                    logger.error(f"Failed to load delta chain for '{name}': {e}. Resetting chain.")
                    deltas = []
            else:
                deltas = []
            
            # 2. Añadir Nueva Operación (Asegurar que tensores en delta_op estén en CPU)
            delta_op_cpu = self.gen.serializer._move_to_cpu(delta_op)
            deltas.append(delta_op_cpu)
            
            # 3. Verificar Umbral de Consolidación Automática
            if len(deltas) >= self.config.delta_consolidation_threshold:
                logger.info(f"Delta chain threshold reached for '{name}'. Initiating automatic consolidation.")
                return self._consolidate(name, old_desc)

            # 4. Serializar (y firmar) y Comprimir Nueva Cadena Delta
            new_deltas_bytes = self.gen.serializer.serialize(deltas)
            compressed_deltas = AdvancedCompressor.compress(new_deltas_bytes, self.config.compression_level)
            
            # 5. Crear Nuevo Descriptor (Versión Incremental)
            # Usamos replace; el checksum se recalcula automáticamente en __post_init__
            new_desc = replace(
                old_desc,
                version=old_desc.version + 1,
                delta_chain=compressed_deltas,
                meta={**old_desc.meta, 'delta_count': len(deltas)}
            )
            
            # 6. Actualizar Tablas y Linaje
            old_addr = ZAddr.compute(old_desc)
            self._register_descriptor(name, new_desc, old_addr)
            
            logger.info(f"Updated '{name}' to version {new_desc.version} (Delta applied).")
            return new_desc

    def _consolidate(self, name: str, desc: ZDescriptor) -> ZDescriptor:
        """
        Consolida la cadena delta sintetizando el estado actual y almacenándolo como una nueva base.
        Debe llamarse dentro de un lock de ZSpace.
        """
        logger.info(f"Consolidating '{name}' (Version {desc.version})...")
        try:
            # 1. Sintetizar el estado actual (requiere cargar fuera del lock si gen.load no es reentrante, 
            # pero aquí asumimos que gen.load maneja su propia concurrencia)
            current_tensor = self.gen.load(desc)
        except Exception as e:
            logger.error(f"Consolidation failed during synthesis for '{name}': {e}")
            return desc # Si falla, retornar el descriptor antiguo

        # 2. Almacenar el tensor sintetizado como nueva base
        # Usar parámetros de descomposición anteriores si están disponibles, o default
        target_ratio = desc.meta.get('decomp_params', {}).get('target_ratio', 0.1)
        new_base_desc = self.gen.store(current_tensor, target_ratio=target_ratio)
        
        # 3. Actualizar Descriptor para reflejar la nueva base (manteniendo versión incremental)
        new_desc = replace(
            new_base_desc,
            version=desc.version + 1,
            meta={**new_base_desc.meta, 'consolidated_from_v': desc.version}
        )

        # 4. Actualizar Tablas y Linaje
        old_addr = ZAddr.compute(desc)
        self._register_descriptor(name, new_desc, old_addr)

        logger.info(f"Consolidated '{name}' to new base version {new_desc.version}.")
        return new_desc

    def get_stats(self) -> Dict:
        """Obtener estadísticas del runtime"""
        return {
            "config": self.config,
            "device": str(self.device),
            "descriptors": len(self.name_to_desc),
            "unique_addresses": len(self.addr_to_desc),
            "performance": self.gen.stats
        }
    
    def cleanup(self):
        """Limpiar recursos."""
        logger.info("Cleaning up MNEME ZSpace...")
        self.gen.shutdown()
        gc.collect()
        if self.device.type.startswith('cuda'):
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        logger.info("Cleanup completed.")

# Alias para compatibilidad
Mneme = ZSpace
