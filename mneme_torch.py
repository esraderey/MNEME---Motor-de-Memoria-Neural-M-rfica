"""
MNEME PyTorch Integration
Integración avanzada con PyTorch para compresión transparente de modelos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import weakref
from mneme_core import ZSpace, ZDescriptor, DecompType, CompressionLevel
import logging
import time
from contextlib import contextmanager
import gc
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

# Instancia global de MNEME
_zspace = ZSpace(cache_size=2 << 30)  # 2GB cache

@dataclass
class CompressionConfig:
    """Configuración de compresión para capas"""
    target_ratio: float = 0.1
    decomp_type: Optional[DecompType] = None
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    memory_limit: Optional[int] = None
    enable_quantization: bool = True
    quantization_bits: int = 8

class ZParameter(nn.Parameter):
    """Parámetro respaldado por síntesis MNEME"""
    
    def __new__(cls, data: torch.Tensor = None, 
                descriptor: ZDescriptor = None,
                requires_grad: bool = True,
                config: CompressionConfig = None):
        if descriptor is not None:
            # Cargar desde descriptor
            data = _zspace.load_desc(descriptor)
        
        instance = super().__new__(cls, data, requires_grad)
        instance._descriptor = descriptor
        instance._zspace_name = None
        instance._config = config or CompressionConfig()
        instance._last_access = time.time()
        return instance
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, 
                   name: str,
                   config: CompressionConfig = None,
                   requires_grad: bool = True):
        """Crear ZParameter desde tensor con descomposición automática"""
        config = config or CompressionConfig()
        desc = _zspace.register(name, tensor, 
                               target_ratio=config.target_ratio,
                               decomp_type=config.decomp_type,
                               memory_limit=config.memory_limit)
        param = cls(descriptor=desc, requires_grad=requires_grad, config=config)
        param._zspace_name = name
        return param
    
    def update_delta(self, delta_op: dict):
        """Aplicar actualización delta y obtener nueva versión"""
        if self._zspace_name:
            self._descriptor = _zspace.update(self._zspace_name, delta_op)
            # Recargar datos
            self.data = _zspace.load_desc(self._descriptor)
            self._last_access = time.time()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de compresión"""
        if self._descriptor:
            return {
                "compression_ratio": self._descriptor.meta.get('compression_ratio', 1.0),
                "decomp_type": self._descriptor.decomp_type.value,
                "version": self._descriptor.version,
                "shape": self._descriptor.shape,
                "size_bytes": len(self._descriptor.seed)
            }
        return {}

class ZLinear(nn.Module):
    """Capa lineal con pesos comprimidos por MNEME"""
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True,
                 config: CompressionConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Configuración de compresión
        self.config = config or CompressionConfig()
        
        # Inicializar peso
        weight = torch.randn(out_features, in_features) / math.sqrt(in_features)
        
        # Registrar con MNEME
        name = f"linear_{id(self)}_weight"
        self.weight = ZParameter.from_tensor(
            weight, name, self.config
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Estadísticas
        self._forward_count = 0
        self._total_time = 0.0
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass con medición de rendimiento"""
        start_time = time.time()
        
        # Actualizar acceso
        self.weight._last_access = time.time()
        
        result = F.linear(input, self.weight, self.bias)
        
        # Actualizar estadísticas
        self._forward_count += 1
        self._total_time += time.time() - start_time
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        avg_time = self._total_time / max(1, self._forward_count)
        compression_stats = self.weight.get_compression_stats()
        
        return {
            "forward_count": self._forward_count,
            "avg_forward_time": avg_time,
            "compression": compression_stats
        }
    
    def extra_repr(self) -> str:
        """Representación adicional con información de compresión"""
        compression_stats = self.weight.get_compression_stats()
        ratio = compression_stats.get('compression_ratio', 1.0)
        decomp_type = compression_stats.get('decomp_type', 'unknown')
        
        return (f'in_features={self.in_features}, '
               f'out_features={self.out_features}, '
               f'bias={self.bias is not None}, '
               f'compression={ratio:.3f}x, '
               f'type={decomp_type}')

class ZConv2d(nn.Module):
    """Convolución 2D con pesos comprimidos por MNEME"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 config: CompressionConfig = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.config = config or CompressionConfig()
        
        # Inicializar peso
        weight = torch.randn(out_channels, in_channels // groups, 
                           kernel_size, kernel_size)
        
        # Registrar con MNEME
        name = f"conv2d_{id(self)}_weight"
        self.weight = ZParameter.from_tensor(weight, name, self.config)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        self.weight._last_access = time.time()
        return F.conv2d(input, self.weight, self.bias, 
                       self.stride, self.padding, self.dilation, self.groups)
    
    def extra_repr(self) -> str:
        compression_stats = self.weight.get_compression_stats()
        ratio = compression_stats.get('compression_ratio', 1.0)
        
        return (f'in_channels={self.in_channels}, '
               f'out_channels={self.out_channels}, '
               f'kernel_size={self.kernel_size}, '
               f'compression={ratio:.3f}x')

class ZAttention(nn.Module):
    """Atención multi-cabeza con compresión MNEME"""
    
    def __init__(self, embed_dim: int, num_heads: int,
                 config: CompressionConfig = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim debe ser divisible por num_heads"
        
        self.config = config or CompressionConfig()
        
        # Proyecciones Q, K, V con compresión
        self.q_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.k_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.v_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.out_proj = ZLinear(embed_dim, embed_dim, config=config)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass de atención"""
        B, L, D = x.shape
        
        # Proyectar y redimensionar
        Q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Atención
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)
        
        out = attn @ V
        
        # Redimensionar y proyectar
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn
        return out

class ZTransformerBlock(nn.Module):
    """Bloque Transformer con compresión MNEME"""
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 mlp_ratio: float = 4.0,
                 config: CompressionConfig = None,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Normalización y atención
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ZAttention(embed_dim, num_heads, config, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP con compresión
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ZLinear(embed_dim, mlp_dim, config=config),
            self._get_activation(activation),
            nn.Dropout(dropout),
            ZLinear(mlp_dim, embed_dim, config=config)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, activation: str):
        """Obtener función de activación"""
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass del bloque Transformer"""
        # Self-attention
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        
        return x

class ZLSTM(nn.Module):
    """LSTM con pesos comprimidos por MNEME"""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1,
                 config: CompressionConfig = None,
                 dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.config = config or CompressionConfig()
        
        # Crear capas LSTM con compresión
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                ZLSTMCell(layer_input_size, hidden_size, config)
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass del LSTM"""
        batch_size = x.size(0)
        
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h_0, c_0)
        
        outputs = []
        h_n, c_n = hidden
        
        for t in range(x.size(1)):
            h_t = []
            c_t = []
            
            for layer in range(self.num_layers):
                if layer == 0:
                    h, c = self.lstm_layers[layer](x[:, t], (h_n[layer], c_n[layer]))
                else:
                    h, c = self.lstm_layers[layer](h, (h_n[layer], c_n[layer]))
                
                h_t.append(h)
                c_t.append(c)
                
                if self.dropout and layer < self.num_layers - 1:
                    h = self.dropout(h)
            
            h_n = torch.stack(h_t)
            c_n = torch.stack(c_t)
            outputs.append(h)
        
        output = torch.stack(outputs, dim=1)
        return output, (h_n, c_n)

class ZLSTMCell(nn.Module):
    """Celda LSTM individual con compresión MNEME"""
    
    def __init__(self, input_size: int, hidden_size: int,
                 config: CompressionConfig = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.config = config or CompressionConfig()
        
        # Pesos de entrada
        self.weight_ih = ZParameter.from_tensor(
            torch.randn(4 * hidden_size, input_size),
            f"lstm_ih_{id(self)}", config
        )
        
        # Pesos de estado oculto
        self.weight_hh = ZParameter.from_tensor(
            torch.randn(4 * hidden_size, hidden_size),
            f"lstm_hh_{id(self)}", config
        )
        
        # Bias
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
    
    def forward(self, input: torch.Tensor, 
                hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass de la celda LSTM"""
        hx, cx = hidden
        
        # Calcular puertas
        gates = F.linear(input, self.weight_ih, self.bias_ih) + \
                F.linear(hx, self.weight_hh, self.bias_hh)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        return hy, cy

def compress_model(model: nn.Module, 
                  config: CompressionConfig = None,
                  min_params: int = 10000,
                  exclude_layers: List[str] = None) -> nn.Module:
    """Comprimir modelo existente reemplazando capas con versiones MNEME"""
    
    config = config or CompressionConfig()
    exclude_layers = exclude_layers or []
    
    def replace_linear(module: nn.Module, prefix: str = ""):
        """Reemplazar capas Lineales con ZLinear"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear) and full_name not in exclude_layers:
                if child.weight.numel() >= min_params:
                    # Reemplazar con ZLinear
                    z_linear = ZLinear(
                        child.in_features,
                        child.out_features,
                        child.bias is not None,
                        config=config
                    )
                    
                    # Copiar pesos
                    with torch.no_grad():
                        z_linear.weight.data = child.weight.data
                        if child.bias is not None:
                            z_linear.bias.data = child.bias.data
                    
                    setattr(module, name, z_linear)
                    logger.info(f"Replaced Linear layer '{full_name}' with ZLinear")
            else:
                replace_linear(child, full_name)
    
    def replace_conv2d(module: nn.Module, prefix: str = ""):
        """Reemplazar capas Conv2d con ZConv2d"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Conv2d) and full_name not in exclude_layers:
                if child.weight.numel() >= min_params:
                    # Reemplazar con ZConv2d
                    z_conv = ZConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        child.stride,
                        child.padding,
                        child.dilation,
                        child.groups,
                        child.bias is not None,
                        config=config
                    )
                    
                    # Copiar pesos
                    with torch.no_grad():
                        z_conv.weight.data = child.weight.data
                        if child.bias is not None:
                            z_conv.bias.data = child.bias.data
                    
                    setattr(module, name, z_conv)
                    logger.info(f"Replaced Conv2d layer '{full_name}' with ZConv2d")
            else:
                replace_conv2d(child, full_name)
    
    # Crear copia del modelo
    model_copy = model
    
    # Reemplazar capas
    replace_linear(model_copy)
    replace_conv2d(model_copy)
    
    return model_copy

def get_compression_stats(model: nn.Module) -> Dict[str, Any]:
    """Obtener estadísticas de compresión del modelo"""
    stats = {
        "original_params": 0,
        "compressed_params": 0,
        "layers": [],
        "compression_ratios": [],
        "total_layers": 0,
        "compressed_layers": 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (ZLinear, ZConv2d, ZAttention)):
            if hasattr(module, 'weight') and hasattr(module.weight, 'get_compression_stats'):
                compression_stats = module.weight.get_compression_stats()
                original = compression_stats.get('size_bytes', 0) / compression_stats.get('compression_ratio', 1.0)
                compressed = compression_stats.get('size_bytes', 0)
                ratio = compression_stats.get('compression_ratio', 1.0)
                
                stats["original_params"] += original
                stats["compressed_params"] += compressed
                stats["compression_ratios"].append(ratio)
                stats["compressed_layers"] += 1
                
                stats["layers"].append({
                    "name": name,
                    "type": type(module).__name__,
                    "original_bytes": original,
                    "compressed_bytes": compressed,
                    "compression_ratio": ratio,
                    "decomp_type": compression_stats.get('decomp_type', 'unknown')
                })
        
        stats["total_layers"] += 1
    
    if stats["original_params"] > 0:
        stats["overall_ratio"] = stats["compressed_params"] / stats["original_params"]
        stats["avg_compression_ratio"] = sum(stats["compression_ratios"]) / len(stats["compression_ratios"]) if stats["compression_ratios"] else 1.0
    
    return stats

def get_model_performance_stats(model: nn.Module) -> Dict[str, Any]:
    """Obtener estadísticas de rendimiento del modelo"""
    stats = {
        "total_forward_time": 0.0,
        "total_forward_count": 0,
        "layers": []
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (ZLinear, ZConv2d, ZAttention)):
            if hasattr(module, 'get_performance_stats'):
                layer_stats = module.get_performance_stats()
                stats["total_forward_time"] += layer_stats.get("avg_forward_time", 0.0) * layer_stats.get("forward_count", 0)
                stats["total_forward_count"] += layer_stats.get("forward_count", 0)
                
                stats["layers"].append({
                    "name": name,
                    "type": type(module).__name__,
                    **layer_stats
                })
    
    if stats["total_forward_count"] > 0:
        stats["avg_forward_time"] = stats["total_forward_time"] / stats["total_forward_count"]
    
    return stats

def optimize_model_memory(model: nn.Module, 
                         target_memory_mb: int = 100,
                         config: CompressionConfig = None) -> nn.Module:
    """Optimizar modelo para uso de memoria específico"""
    config = config or CompressionConfig()
    
    # Calcular compresión necesaria
    current_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    target_ratio = target_memory_mb / current_memory
    
    if target_ratio < 1.0:
        # Ajustar configuración para mayor compresión
        config.target_ratio = min(target_ratio * 0.8, 0.05)  # 5% mínimo
        config.compression_level = CompressionLevel.MAXIMUM
        
        # Comprimir modelo
        compressed_model = compress_model(model, config)
        
        # Verificar memoria resultante
        new_memory = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
        logger.info(f"Memory optimization: {current_memory:.1f}MB -> {new_memory:.1f}MB")
        
        return compressed_model
    
    return model

# Alias para compatibilidad
MLinear = ZLinear
MConv2d = ZConv2d
MAttention = ZAttention
MTransformerBlock = ZTransformerBlock