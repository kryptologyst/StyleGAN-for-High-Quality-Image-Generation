"""
StyleGAN Generator implementation with proper architecture.

This module contains the StyleGAN generator with mapping network,
style modulation, noise injection, and synthesis blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class EqualizedLinear(nn.Module):
    """Equalized learning rate linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # Scale weights by 1/sqrt(fan_in)
        fan_in = in_features
        self.scale = np.sqrt(2.0 / fan_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.scale
        return F.linear(x, weight, self.bias)


class EqualizedConv2d(nn.Module):
    """Equalized learning rate 2D convolution."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        self.stride = stride
        self.padding = padding
        
        # Scale weights by 1/sqrt(fan_in)
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = np.sqrt(2.0 / fan_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.scale
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class MappingNetwork(nn.Module):
    """Mapping network that transforms z to w."""
    
    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_layers: int = 8) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(EqualizedLinear(z_dim, w_dim))
            else:
                layers.append(EqualizedLinear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Transform z to w."""
        return self.network(z)


class NoiseInjection(nn.Module):
    """Noise injection layer."""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to the input."""
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        
        return x + self.weight * noise


class StyleModulation(nn.Module):
    """Style modulation (AdaIN) layer."""
    
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.w_dim = w_dim
        
        # Style modulation layers
        self.style_scale = EqualizedLinear(w_dim, channels)
        self.style_bias = EqualizedLinear(w_dim, channels)
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply style modulation."""
        # Normalize x
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std
        
        # Apply style
        style_scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        style_bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        
        return style_scale * x_norm + style_bias


class StyleGANBlock(nn.Module):
    """StyleGAN synthesis block."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        w_dim: int, 
        resolution: int,
        upsampling: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.upsampling = upsampling
        
        # Upsampling
        if upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()
        
        # Convolution layers
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        
        # Style modulation
        self.style1 = StyleModulation(out_channels, w_dim)
        self.style2 = StyleModulation(out_channels, w_dim)
        
        # Noise injection
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        # Upsample if needed
        x = self.upsample(x)
        
        # First conv + style + noise + activation
        x = self.conv1(x)
        x = self.style1(x, w)
        x = self.noise1(x)
        x = self.activation(x)
        
        # Second conv + style + noise + activation
        x = self.conv2(x)
        x = self.style2(x, w)
        x = self.noise2(x)
        x = self.activation(x)
        
        return x


class StyleGANGenerator(nn.Module):
    """StyleGAN Generator with proper architecture."""
    
    def __init__(
        self, 
        z_dim: int = 512, 
        w_dim: int = 512, 
        img_channels: int = 3,
        img_size: int = 64,
        mapping_layers: int = 8
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        
        # Constant input
        self.const = nn.Parameter(torch.randn(1, w_dim, 4, 4))
        
        # Synthesis network
        self.blocks = nn.ModuleList()
        
        # Calculate number of blocks needed
        current_size = 4
        current_channels = w_dim
        
        while current_size < img_size:
            next_size = min(current_size * 2, img_size)
            next_channels = max(current_channels // 2, img_channels)
            
            self.blocks.append(StyleGANBlock(
                current_channels, 
                next_channels, 
                w_dim, 
                next_size,
                upsampling=(next_size > current_size)
            ))
            
            current_size = next_size
            current_channels = next_channels
        
        # To RGB layer
        self.to_rgb = EqualizedConv2d(current_channels, img_channels, 1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise."""
        batch_size = z.shape[0]
        
        # Map z to w
        w = self.mapping(z)
        
        # Start with constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Apply synthesis blocks
        for block in self.blocks:
            x = block(x, w)
        
        # Convert to RGB
        x = self.to_rgb(x)
        
        return torch.tanh(x)
