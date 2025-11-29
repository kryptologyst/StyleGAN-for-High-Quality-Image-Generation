"""
Unit tests for StyleGAN models and utilities.

This module contains tests for the generator, discriminator,
and utility functions.
"""

import pytest
import torch
import numpy as np

from src.models.generator import StyleGANGenerator, MappingNetwork, StyleModulation, NoiseInjection
from src.models.discriminator import StyleGANDiscriminator, MinibatchStdDev
from src.training.trainer import EMA, gradient_penalty
from src.utils.device import set_seed, get_device


class TestStyleGANGenerator:
    """Test cases for StyleGAN Generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = StyleGANGenerator(z_dim=512, img_size=64)
        assert generator.z_dim == 512
        assert generator.img_size == 64
        assert generator.img_channels == 3
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = StyleGANGenerator(z_dim=512, img_size=64)
        batch_size = 4
        z = torch.randn(batch_size, 512)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 3, 64, 64)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # tanh output
    
    def test_mapping_network(self):
        """Test mapping network."""
        mapping = MappingNetwork(z_dim=512, w_dim=512)
        z = torch.randn(4, 512)
        
        w = mapping(z)
        
        assert w.shape == (4, 512)
    
    def test_style_modulation(self):
        """Test style modulation layer."""
        style_mod = StyleModulation(channels=64, w_dim=512)
        x = torch.randn(4, 64, 32, 32)
        w = torch.randn(4, 512)
        
        output = style_mod(x, w)
        
        assert output.shape == x.shape
    
    def test_noise_injection(self):
        """Test noise injection layer."""
        noise_inj = NoiseInjection(channels=64)
        x = torch.randn(4, 64, 32, 32)
        
        output = noise_inj(x)
        
        assert output.shape == x.shape


class TestStyleGANDiscriminator:
    """Test cases for StyleGAN Discriminator."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = StyleGANDiscriminator(img_size=64)
        assert discriminator.img_size == 64
        assert discriminator.img_channels == 3
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = StyleGANDiscriminator(img_size=64)
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_minibatch_std_dev(self):
        """Test minibatch standard deviation layer."""
        minibatch_std = MinibatchStdDev(group_size=4)
        x = torch.randn(8, 64, 32, 32)
        
        output = minibatch_std(x)
        
        assert output.shape == (8, 65, 32, 32)  # +1 channel for std


class TestTrainingUtilities:
    """Test cases for training utilities."""
    
    def test_ema_initialization(self):
        """Test EMA initialization."""
        model = StyleGANGenerator(z_dim=512, img_size=64)
        ema = EMA(model, decay=0.999)
        
        assert len(ema.shadow) > 0
        assert ema.decay == 0.999
    
    def test_ema_update(self):
        """Test EMA update."""
        model = StyleGANGenerator(z_dim=512, img_size=64)
        ema = EMA(model, decay=0.999)
        
        # Update model parameters
        for param in model.parameters():
            param.data += 1.0
        
        ema.update(model)
        
        # Check that shadow parameters are updated
        for name, param in model.named_parameters():
            assert name in ema.shadow
    
    def test_gradient_penalty(self):
        """Test gradient penalty calculation."""
        discriminator = StyleGANDiscriminator(img_size=64)
        real_samples = torch.randn(4, 3, 64, 64)
        fake_samples = torch.randn(4, 3, 64, 64)
        device = torch.device("cpu")
        
        penalty = gradient_penalty(discriminator, real_samples, fake_samples, device)
        
        assert penalty.item() >= 0
        assert penalty.requires_grad


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate two random numbers
        torch.manual_seed(42)
        r1 = torch.randn(1)
        
        set_seed(42)
        torch.manual_seed(42)
        r2 = torch.randn(1)
        
        assert torch.allclose(r1, r2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__])
