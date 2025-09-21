"""
MNEME Usage Examples
Ejemplos prácticos del sistema MNEME con todas las funcionalidades avanzadas
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any, List

# Importar módulos MNEME
from mneme_core import ZSpace, DecompType, CompressionLevel
from mneme_torch import (
    ZLinear, ZConv2d, ZAttention, ZTransformerBlock, 
    compress_model, get_compression_stats, get_model_performance_stats,
    CompressionConfig, optimize_model_memory
)
from mneme_security import SecurityManager, SecurityLevel, AuditEvent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Uso básico de MNEME con todas las funcionalidades"""
    print("="*60)
    print("EJEMPLO BÁSICO DE MNEME")
    print("="*60)
    
    # Inicializar MNEME con configuración avanzada
    mneme = ZSpace(
        cache_size=1 << 30,  # 1GB
        compression_level=CompressionLevel.HIGH,
        enable_merkle=True,
        enable_checksums=True
    )
    
    # Crear tensor de ejemplo
    tensor = torch.randn(1024, 1024)
    print(f"Tensor original: {tensor.shape}, {tensor.numel() * 4 / 1024:.1f}KB")
    
    # Registrar con diferentes configuraciones de compresión
    configs = [
        {"target_ratio": 0.1, "decomp_type": DecompType.TT},
        {"target_ratio": 0.05, "decomp_type": DecompType.SVD},
        {"target_ratio": 0.2, "decomp_type": DecompType.CP}
    ]
    
    for i, config in enumerate(configs):
        desc = mneme.register(f"tensor_{i}", tensor, **config)
        loaded = mneme.load(f"tensor_{i}")
        
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)
        
        print(f"Config {i+1}: Ratio={ratio:.3f}, Error={error:.6f}, "
              f"Tipo={desc.decomp_type.value}")
    
    # Estadísticas del sistema
    stats = mneme.get_stats()
    print(f"\nEstadísticas del sistema:")
    print(f"- Descriptores: {stats['descriptors']}")
    print(f"- Versiones: {stats['versions']}")
    print(f"- Cache hits: {stats['performance']['cache']['hits']}")
    print(f"- Cache misses: {stats['performance']['cache']['misses']}")

def example_advanced_compression():
    """Ejemplo de compresión avanzada con diferentes algoritmos"""
    print("\n" + "="*60)
    print("COMPRESIÓN AVANZADA")
    print("="*60)
    
    mneme = ZSpace(cache_size=512 << 20)  # 512MB
    
    # Crear diferentes tipos de tensores
    tensors = {
        "sparse": torch.zeros(1000, 1000),
        "dense": torch.randn(500, 500),
        "low_rank": torch.randn(200, 200) @ torch.randn(200, 200).T,
        "high_dim": torch.randn(50, 50, 50, 50)
    }
    
    # Hacer sparse el tensor sparse
    tensors["sparse"][::10, ::10] = torch.randn(100, 100)
    
    for name, tensor in tensors.items():
        print(f"\nProcesando tensor {name}: {tensor.shape}")
        
        # Auto-selección de descomposición
        desc = mneme.register(f"{name}_auto", tensor, target_ratio=0.1)
        loaded = mneme.load(f"{name}_auto")
        
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)
        
        print(f"  Auto-selección: {desc.decomp_type.value}, "
              f"Ratio={ratio:.3f}, Error={error:.6f}")
        
        # Probar diferentes tipos de descomposición
        for decomp_type in [DecompType.TT, DecompType.CP, DecompType.TUCKER]:
            try:
                desc = mneme.register(f"{name}_{decomp_type.value}", tensor, 
                                    decomp_type=decomp_type, target_ratio=0.1)
                loaded = mneme.load(f"{name}_{decomp_type.value}")
                
                error = torch.norm(tensor - loaded) / torch.norm(tensor)
                ratio = desc.meta.get('compression_ratio', 1.0)
                
                print(f"  {decomp_type.value}: Ratio={ratio:.3f}, Error={error:.6f}")
            except Exception as e:
                print(f"  {decomp_type.value}: Error - {e}")

def example_model_compression():
    """Compresión de modelos con MNEME"""
    print("\n" + "="*60)
    print("COMPRESIÓN DE MODELOS")
    print("="*60)
    
    # Crear modelo complejo
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = ComplexModel()
    print(f"Modelo original: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Configuración de compresión
    config = CompressionConfig(
        target_ratio=0.1,
        compression_level=CompressionLevel.HIGH,
        memory_limit=50 * 1024 * 1024  # 50MB
    )
    
    # Comprimir modelo
    compressed_model = compress_model(model, config=config, min_params=1000)
    
    # Estadísticas de compresión
    stats = get_compression_stats(compressed_model)
    print(f"\nEstadísticas de compresión:")
    print(f"- Parámetros originales: {stats['original_params']:,.0f}")
    print(f"- Parámetros comprimidos: {stats['compressed_params']:,.0f}")
    print(f"- Ratio general: {stats['overall_ratio']:.3f}")
    print(f"- Capas comprimidas: {stats['compressed_layers']}/{stats['total_layers']}")
    print(f"- Ratio promedio: {stats['avg_compression_ratio']:.3f}")
    
    # Probar inferencia
    x = torch.randn(4, 3, 32, 32)
    
    with torch.no_grad():
        start_time = time.time()
        original_out = model(x)
        original_time = time.time() - start_time
        
        start_time = time.time()
        compressed_out = compressed_model(x)
        compressed_time = time.time() - start_time
    
    diff = torch.norm(original_out - compressed_out) / torch.norm(original_out)
    print(f"\nInferencia:")
    print(f"- Tiempo original: {original_time:.4f}s")
    print(f"- Tiempo comprimido: {compressed_time:.4f}s")
    print(f"- Diferencia de salida: {diff:.6f}")

def example_transformer_compression():
    """Compresión de modelo Transformer"""
    print("\n" + "="*60)
    print("COMPRESIÓN DE TRANSFORMER")
    print("="*60)
    
    # Crear Transformer simple
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=512, num_heads=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
            
            self.layers = nn.ModuleList([
                ZTransformerBlock(embed_dim, num_heads, config=CompressionConfig(target_ratio=0.1))
                for _ in range(num_layers)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.output = nn.Linear(embed_dim, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:x.size(1)]
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.output(x)
    
    model = SimpleTransformer()
    print(f"Transformer original: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Estadísticas de compresión
    stats = get_compression_stats(model)
    print(f"Estadísticas de compresión:")
    print(f"- Parámetros originales: {stats['original_params']:,.0f}")
    print(f"- Parámetros comprimidos: {stats['compressed_params']:,.0f}")
    print(f"- Ratio general: {stats['overall_ratio']:.3f}")
    
    # Probar inferencia
    x = torch.randint(0, 1000, (4, 128))
    
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        inference_time = time.time() - start_time
    
    print(f"Inferencia: {inference_time:.4f}s")
    print(f"Salida: {output.shape}")

def example_security_features():
    """Ejemplo de funcionalidades de seguridad"""
    print("\n" + "="*60)
    print("FUNCIONALIDADES DE SEGURIDAD")
    print("="*60)
    
    # Inicializar gestor de seguridad
    security_manager = SecurityManager(
        security_level=SecurityLevel.HIGH,
        audit_log_file="mneme_audit.log"
    )
    
    # Crear descriptor seguro
    data = torch.randn(100, 100).numpy().tobytes()
    secure_desc = security_manager.create_secure_descriptor(data, "test_resource")
    
    print(f"Descriptor seguro creado:")
    print(f"- Tamaño de datos: {len(data)} bytes")
    print(f"- Checksum: {secure_desc.checksum.hex()[:16]}...")
    print(f"- Firma: {secure_desc.signature.hex()[:16]}...")
    print(f"- Merkle root: {secure_desc.merkle_root.hex()[:16] if secure_desc.merkle_root else 'N/A'}...")
    
    # Verificar integridad
    integrity_ok = secure_desc.verify_integrity()
    signature_ok = secure_desc.verify_signature()
    
    print(f"\nVerificaciones:")
    print(f"- Integridad: {'✓' if integrity_ok else '✗'}")
    print(f"- Firma: {'✓' if signature_ok else '✗'}")
    
    # Obtener prueba de Merkle
    if secure_desc.merkle_root:
        proof = secure_desc.get_merkle_proof(0)
        proof_ok = secure_desc.verify_merkle_proof(proof, 0)
        print(f"- Prueba Merkle: {'✓' if proof_ok else '✗'}")
    
    # Estado de seguridad
    security_status = security_manager.get_security_status()
    print(f"\nEstado de seguridad:")
    print(f"- Nivel: {security_status['security_level']}")
    print(f"- Recursos bloqueados: {len(security_status['locked_resources'])}")
    print(f"- Intentos fallidos: {security_status['failed_attempts']}")
    print(f"- Sesiones activas: {security_status['active_sessions']}")

def example_memory_optimization():
    """Ejemplo de optimización de memoria"""
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DE MEMORIA")
    print("="*60)
    
    # Crear modelo grande
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(1000, 1000) for _ in range(10)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x
    
    model = LargeModel()
    original_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Memoria original: {original_memory:.1f}MB")
    
    # Optimizar para memoria específica
    target_memory = 50  # 50MB
    optimized_model = optimize_model_memory(model, target_memory_mb=target_memory)
    
    new_memory = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / (1024 * 1024)
    print(f"Memoria optimizada: {new_memory:.1f}MB")
    print(f"Reducción: {(1 - new_memory/original_memory)*100:.1f}%")
    
    # Estadísticas de rendimiento
    x = torch.randn(32, 1000)
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = optimized_model(x)
        avg_time = (time.time() - start_time) / 100
    
    print(f"Tiempo promedio de inferencia: {avg_time*1000:.2f}ms")

def example_incremental_updates():
    """Ejemplo de actualizaciones incrementales"""
    print("\n" + "="*60)
    print("ACTUALIZACIONES INCREMENTALES")
    print("="*60)
    
    mneme = ZSpace()
    
    # Estado inicial
    state = torch.zeros(100, 100)
    desc = mneme.register("game_state", state)
    print(f"Estado inicial: versión {desc.version}")
    
    # Simular actualizaciones del juego
    for step in range(10):
        # Actualización dispersa
        indices = torch.randint(0, 100, (20, 2))
        values = torch.randn(20)
        
        delta = {
            "type": "sparse_update",
            "indices": indices,
            "values": values
        }
        
        desc = mneme.update("game_state", delta)
        print(f"Paso {step+1}: versión {desc.version}, {len(indices)} actualizaciones")
    
    # Cargar estado final
    final_state = mneme.load("game_state")
    non_zero = (final_state != 0).sum().item()
    print(f"\nEstado final: {non_zero} elementos no cero")
    
    # Estadísticas de versiones
    stats = mneme.get_stats()
    print(f"Versiones creadas: {stats['versions']}")

def example_performance_benchmark():
    """Benchmark de rendimiento"""
    print("\n" + "="*60)
    print("BENCHMARK DE RENDIMIENTO")
    print("="*60)
    
    mneme = ZSpace(cache_size=1 << 30)
    
    # Diferentes tamaños de tensor
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for size in sizes:
        print(f"\nTamaño: {size}")
        
        # Crear tensor
        tensor = torch.randn(size)
        original_size = tensor.numel() * 4 / 1024  # KB
        
        # Medir tiempo de almacenamiento
        start_time = time.time()
        desc = mneme.register(f"tensor_{size}", tensor)
        store_time = time.time() - start_time
        
        # Medir tiempo de carga
        start_time = time.time()
        loaded = mneme.load(f"tensor_{size}")
        load_time = time.time() - start_time
        
        # Verificar precisión
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)
        
        print(f"  Tamaño original: {original_size:.1f}KB")
        print(f"  Tamaño comprimido: {len(desc.seed)/1024:.1f}KB")
        print(f"  Ratio: {ratio:.3f}")
        print(f"  Error: {error:.6f}")
        print(f"  Tiempo almacenamiento: {store_time*1000:.2f}ms")
        print(f"  Tiempo carga: {load_time*1000:.2f}ms")

def main():
    """Función principal con todos los ejemplos"""
    print("🧠 MNEME - Motor de Memoria Neural Mórfica")
    print("Ejemplos de uso avanzado\n")
    
    try:
        example_basic_usage()
        example_advanced_compression()
        example_model_compression()
        example_transformer_compression()
        example_security_features()
        example_memory_optimization()
        example_incremental_updates()
        example_performance_benchmark()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error en ejemplos: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()