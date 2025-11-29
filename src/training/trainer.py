"""
Training utilities for StyleGAN.

This module contains training functions, EMA, gradient penalty,
and other training stabilization techniques.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from typing import Tuple


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module) -> None:
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def gradient_penalty(
    discriminator: nn.Module, 
    real_samples: torch.Tensor, 
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


def save_samples(
    generator: nn.Module, 
    epoch: int, 
    num_samples: int = 64,
    device: torch.device = None,
    save_dir: str = "assets/samples"
) -> None:
    """Save generated samples."""
    if device is None:
        device = next(generator.parameters()).device
    
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.z_dim, device=device)
        fake_images = generator(z).cpu()
        
        # Create grid
        grid_img = torchvision.utils.make_grid(
            fake_images, 
            nrow=8, 
            normalize=True, 
            value_range=(-1, 1)
        )
        
        # Save image
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(
            grid_img, 
            os.path.join(save_dir, f"epoch_{epoch:04d}.png")
        )
    
    generator.train()


def train_stylegan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    num_epochs: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    lambda_gp: float = 10.0,
    d_steps: int = 1,
    g_steps: int = 1,
    device: torch.device = None,
    save_samples_every: int = 10,
    log_every_n_steps: int = 100
) -> None:
    """Train StyleGAN with proper stabilization techniques."""
    
    if device is None:
        device = next(generator.parameters()).device
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Create optimizers
    optimizer_g = optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(beta1, beta2)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(), 
        lr=lr, 
        betas=(beta1, beta2)
    )
    
    # Create EMA
    ema = EMA(generator, decay=0.999)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            for _ in range(d_steps):
                optimizer_d.zero_grad()
                
                # Real images
                real_pred = discriminator(real_images)
                d_loss_real = -real_pred.mean()
                
                # Fake images
                z = torch.randn(batch_size, generator.z_dim, device=device)
                fake_images = generator(z).detach()
                fake_pred = discriminator(fake_images)
                d_loss_fake = fake_pred.mean()
                
                # Gradient penalty
                gp = gradient_penalty(discriminator, real_images, fake_images, device)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake + lambda_gp * gp
                d_loss.backward()
                optimizer_d.step()
            
            # Train Generator
            for _ in range(g_steps):
                optimizer_g.zero_grad()
                
                z = torch.randn(batch_size, generator.z_dim, device=device)
                fake_images = generator(z)
                fake_pred = discriminator(fake_images)
                g_loss = -fake_pred.mean()
                
                g_loss.backward()
                optimizer_g.step()
                
                # Update EMA
                ema.update(generator)
            
            # Print progress
            if batch_idx % log_every_n_steps == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, GP: {gp.item():.4f}')
        
        # Save samples
        if (epoch + 1) % save_samples_every == 0:
            save_samples(generator, epoch + 1, device=device)
            print(f'Saved samples for epoch {epoch + 1}')
