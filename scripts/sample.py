#!/usr/bin/env python3
"""
Sampling script for StyleGAN.

This script generates samples from a trained StyleGAN model
with various controls for seed, number of samples, etc.
"""

import argparse
import torch
import torchvision
import yaml
from pathlib import Path

from src.models.generator import StyleGANGenerator
from src.utils.device import set_seed, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_samples(
    generator: StyleGANGenerator,
    num_samples: int = 64,
    seed: int = None,
    device: torch.device = None,
    save_path: str = "samples.png"
) -> None:
    """Generate and save samples from the generator."""
    if device is None:
        device = next(generator.parameters()).device
    
    generator.eval()
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        
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
        torchvision.utils.save_image(grid_img, save_path)
        print(f"Saved {num_samples} samples to {save_path}")


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Generate samples from StyleGAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--num-samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for generation')
    parser.add_argument('--output', type=str, default='samples.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed and device
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(config['device']['seed'])
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create generator
    generator = StyleGANGenerator(
        z_dim=config['model']['z_dim'],
        w_dim=config['model']['w_dim'],
        img_channels=config['model']['img_channels'],
        img_size=config['model']['img_size'],
        mapping_layers=config['model']['mapping_layers']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator = generator.to(device)
    
    # Generate samples
    generate_samples(
        generator=generator,
        num_samples=args.num_samples,
        seed=args.seed,
        device=device,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
