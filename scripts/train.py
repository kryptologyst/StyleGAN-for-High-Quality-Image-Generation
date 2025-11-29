#!/usr/bin/env python3
"""
Main training script for StyleGAN.

This script provides a clean interface for training StyleGAN models
with proper configuration management and logging.
"""

import argparse
import yaml
from pathlib import Path

from src.models.generator import StyleGANGenerator
from src.models.discriminator import StyleGANDiscriminator
from src.data.dataloader import create_data_loaders
from src.training.trainer import train_stylegan
from src.utils.device import set_seed, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train StyleGAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Override dataset name')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset is not None:
        config['data']['dataset'] = args.dataset
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.seed is not None:
        config['device']['seed'] = args.seed
    
    # Set seed and device
    set_seed(config['device']['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    # Create models
    generator = StyleGANGenerator(
        z_dim=config['model']['z_dim'],
        w_dim=config['model']['w_dim'],
        img_channels=config['model']['img_channels'],
        img_size=config['model']['img_size'],
        mapping_layers=config['model']['mapping_layers']
    )
    
    discriminator = StyleGANDiscriminator(
        img_channels=config['model']['img_channels'],
        img_size=config['model']['img_size']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_name=config['data']['dataset'],
        batch_size=config['training']['batch_size'],
        img_size=config['model']['img_size'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Training StyleGAN on {config['data']['dataset']} dataset")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train the model
    train_stylegan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['lr'],
        beta1=config['training']['beta1'],
        beta2=config['training']['beta2'],
        lambda_gp=config['training']['lambda_gp'],
        d_steps=config['training']['d_steps'],
        g_steps=config['training']['g_steps'],
        device=device,
        save_samples_every=config['evaluation']['save_samples_every'],
        log_every_n_steps=config['logging']['log_every_n_steps']
    )


if __name__ == "__main__":
    main()
