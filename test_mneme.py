"""
MNEME Test Suite
Suite de pruebas para verificar el funcionamiento de MNEME
"""

import torch
import numpy as np
import time
import logging
import unittest

# Importar módulos MNEME
from mneme_core import ZSpace, DecompType, CompressionLevel
from mneme_torch import ZLinear, compress_model, get_compression_stats, CompressionConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMNEMECore(unittest.TestCase):
    """Pruebas del núcleo de MNEME"""
    
    def setUp(self):
        """Configurar para cada prueba"""
        self.mneme = ZSpace(cache_size=100 << 20)  # 100MB cache
    
    def test_basic_tensor_operations(self):
        """Probar operaciones básicas con tensores"""
        # Crear tensor de prueba
        tensor = torch.randn(100, 100)
        
        # Registrar tensor
        desc = self.mneme.register("test_tensor", tensor, target_ratio=0.1)
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape, tensor.shape)
        
        # Cargar tensor
        loaded = self.mneme.load("test_tensor")
        self.assertEqual(loaded.shape, tensor.shape)
        
        # Verificar precisión
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        self.assertLess(error, 0.01)  # Error < 1%
    
    def test_different_decomp_types(self):
        """Probar diferentes tipos de descomposición"""
        tensor = torch.randn(50, 50)
        
        decomp_types = [DecompType.TT, DecompType.CP, DecompType.SVD]
        
        for decomp_type in decomp_types:
            with self.subTest(decomp_type=decomp_type):
                desc = self.mneme.register(
                    f"test_{decomp_type.value}", 
                    tensor, 
                    decomp_type=decomp_type,
                    target_ratio=0.1
                )
                
                loaded = self.mneme.load(f"test_{decomp_type.value}")
                error = torch.norm(tensor - loaded) / torch.norm(tensor)
                
                self.assertLess(error, 0.1)  # Error < 10%

class TestMNEMETorch(unittest.TestCase):
    """Pruebas de integración con PyTorch"""
    
    def test_zlinear_layer(self):
        """Probar capa ZLinear"""
        layer = ZLinear(100, 50, config=CompressionConfig(target_ratio=0.1))
        
        # Forward pass
        x = torch.randn(32, 100)
        output = layer(x)
        
        self.assertEqual(output.shape, (32, 50))
    
    def test_model_compression(self):
        """Probar compresión de modelo"""
        # Crear modelo simple
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        
        # Comprimir modelo
        config = CompressionConfig(target_ratio=0.1)
        compressed_model = compress_model(model, config=config)
        
        # Verificar que la compresión funcionó
        stats = get_compression_stats(compressed_model)
        self.assertGreater(stats["compressed_layers"], 0)
        self.assertLess(stats["overall_ratio"], 1.0)

def run_performance_benchmark():
    """Ejecutar benchmark de rendimiento"""
    print("\n" + "="*60)
    print("BENCHMARK DE RENDIMIENTO")
    print("="*60)
    
    # Configurar MNEME
    mneme = ZSpace(cache_size=1 << 30)  # 1GB
    
    # Diferentes tamaños de tensor
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        print(f"\nTamaño: {size}")
        
        # Crear tensor
        tensor = torch.randn(size)
        original_size = tensor.numel() * 4 / 1024  # KB
        
        # Medir tiempo de almacenamiento
        start_time = time.time()
        desc = mneme.register(f"benchmark_{size}", tensor)
        store_time = time.time() - start_time
        
        # Medir tiempo de carga
        start_time = time.time()
        loaded = mneme.load(f"benchmark_{size}")
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
    """Función principal de pruebas"""
    print("🧠 MNEME - Suite de Pruebas")
    print("="*60)
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Ejecutar benchmark de rendimiento
    run_performance_benchmark()
    
    print("\n✅ Todas las pruebas completadas")

if __name__ == "__main__":
    main()