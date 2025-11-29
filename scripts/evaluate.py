#!/usr/bin/env python3
"""
Evaluation script for StyleGAN.

This script evaluates a trained StyleGAN model using various metrics
including FID, KID, Precision, and Recall.
"""

import argparse
import torch
import yaml
import json
from pathlib import Path

from src.models.generator import StyleGANGenerator
from src.data.dataloader import create_data_loaders
from src.evaluation.metrics import evaluate_generator
from src.utils.device import get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate StyleGAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples for evaluation')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
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
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_name=config['data']['dataset'],
        batch_size=args.batch_size,
        img_size=config['model']['img_size'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Evaluating StyleGAN on {config['data']['dataset']} dataset")
    print(f"Number of samples: {args.num_samples}")
    
    # Evaluate generator
    results = evaluate_generator(
        generator=generator,
        real_loader=val_loader,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"FID: {results['fid']:.4f}")
    print(f"KID: {results['kid']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
