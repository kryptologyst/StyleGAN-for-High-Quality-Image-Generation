"""
StyleGAN Discriminator implementation with proper architecture.

This module contains the StyleGAN discriminator with minibatch
standard deviation and proper stabilization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for improved training stability."""
    
    def __init__(self, group_size: int = 4, num_channels: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add minibatch standard deviation."""
        batch_size, channels, height, width = x.shape
        
        # Group samples
        group_size = min(self.group_size, batch_size)
        if batch_size % group_size != 0:
            group_size = batch_size
        
        # Reshape to groups
        grouped = x.view(group_size, -1, channels, height, width)
        
        # Calculate std dev
        std = grouped.std(dim=0, keepdim=True)
        std = std.mean(dim=[2, 3, 4], keepdim=True)
        std = std.expand(group_size, self.num_channels, height, width)
        
        # Concatenate
        return torch.cat([x, std], dim=1)


class DiscriminatorBlock(nn.Module):
    """Discriminator block with downsampling."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        resolution: int,
        downsampling: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.downsampling = downsampling
        
        # Convolution layers
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        
        # Downsampling
        if downsampling:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = nn.Identity()
        
        # Skip connection
        if in_channels != out_channels or downsampling:
            self.skip = EqualizedConv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        # Main path
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.downsample(out)
        
        # Skip connection
        skip = self.skip(x)
        skip = self.downsample(skip)
        
        return self.activation(out + skip)


class StyleGANDiscriminator(nn.Module):
    """StyleGAN Discriminator with proper architecture."""
    
    def __init__(
        self, 
        img_channels: int = 3, 
        img_size: int = 64,
        use_minibatch_std: bool = True
    ) -> None:
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.use_minibatch_std = use_minibatch_std
        
        # Build discriminator blocks
        self.blocks = nn.ModuleList()
        
        # Calculate number of blocks needed
        current_size = img_size
        current_channels = img_channels
        
        while current_size > 4:
            next_size = max(current_size // 2, 4)
            next_channels = min(current_channels * 2, 512)
            
            self.blocks.append(DiscriminatorBlock(
                current_channels, 
                next_channels, 
                current_size,
                downsampling=(next_size < current_size)
            ))
            
            current_size = next_size
            current_channels = next_channels
        
        # Minibatch standard deviation
        if use_minibatch_std:
            self.minibatch_std = MinibatchStdDev()
            current_channels += 1
        else:
            self.minibatch_std = nn.Identity()
        
        # Final layers
        self.final_conv = EqualizedConv2d(current_channels, current_channels, 3, padding=1)
        self.final_linear = EqualizedLinear(current_channels * 4 * 4, 1)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        # Apply discriminator blocks
        for block in self.blocks:
            x = block(x)
        
        # Minibatch standard deviation
        x = self.minibatch_std(x)
        
        # Final layers
        x = self.final_conv(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        
        return x
