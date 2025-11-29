"""
StyleGAN for High-Quality Image Generation

StyleGAN (Style Generative Adversarial Network) is a state-of-the-art generative model 
known for generating high-quality, photorealistic images. StyleGAN introduces a 
style-based generator architecture that allows for fine-grained control over the 
generated images, making it possible to manipulate features like texture, color, 
and overall style at various layers of the network.

This implementation provides a modern, production-ready StyleGAN with proper
architecture, training stabilization, and evaluation metrics.
"""

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Set seed for reproducibility
set_seed(42)
device = get_device()
print(f"Using device: {device}")
 
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


def create_data_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 64,
    img_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        val_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def save_samples(
    generator: nn.Module, 
    epoch: int, 
    num_samples: int = 64,
    device: torch.device = device
) -> None:
    """Save generated samples."""
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
        os.makedirs("assets/samples", exist_ok=True)
        torchvision.utils.save_image(
            grid_img, 
            f"assets/samples/epoch_{epoch:04d}.png"
        )
    
    generator.train()


def train_stylegan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    lambda_gp: float = 10.0,
    d_steps: int = 1,
    g_steps: int = 1,
    device: torch.device = device
) -> None:
    """Train StyleGAN with proper stabilization techniques."""
    
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
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, GP: {gp.item():.4f}')
        
        # Save samples
        if (epoch + 1) % 10 == 0:
            save_samples(generator, epoch + 1, device=device)
            print(f'Saved samples for epoch {epoch + 1}')


# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        "z_dim": 512,
        "w_dim": 512,
        "img_size": 64,
        "img_channels": 3,
        "batch_size": 64,
        "num_epochs": 100,
        "lr": 0.0002,
        "lambda_gp": 10.0,
        "dataset": "cifar10"
    }
    
    # Create models
    generator = StyleGANGenerator(
        z_dim=config["z_dim"],
        w_dim=config["w_dim"],
        img_channels=config["img_channels"],
        img_size=config["img_size"]
    )
    
    discriminator = StyleGANDiscriminator(
        img_channels=config["img_channels"],
        img_size=config["img_size"]
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        img_size=config["img_size"]
    )
    
    print(f"Training StyleGAN on {config['dataset']} dataset")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train the model
    train_stylegan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        lambda_gp=config["lambda_gp"],
        device=device
    )