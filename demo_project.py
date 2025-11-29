#!/usr/bin/env python3
"""
StyleGAN Project Summary and Demo

This script provides a quick demonstration of the StyleGAN implementation
and shows the project structure and capabilities.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.generator import StyleGANGenerator
from models.discriminator import StyleGANDiscriminator
from data.dataloader import create_data_loaders
from training.trainer import save_samples
from utils.device import set_seed, get_device


def print_project_info():
    """Print project information and structure."""
    print("=" * 60)
    print("🎨 StyleGAN for High-Quality Image Generation")
    print("=" * 60)
    print()
    
    print("📁 Project Structure:")
    structure = """
    stylegan-high-quality-generation/
    ├── src/                    # Source code
    │   ├── models/            # Model implementations
    │   ├── training/          # Training utilities
    │   ├── data/              # Data handling
    │   ├── evaluation/         # Evaluation metrics
    │   └── utils/             # Utilities
    ├── configs/               # Configuration files
    ├── scripts/               # Training and sampling scripts
    ├── demo/                  # Streamlit demo app
    ├── assets/               # Generated assets
    ├── tests/                # Unit tests
    └── notebooks/            # Jupyter notebooks
    """
    print(structure)
    
    print("🚀 Key Features:")
    features = [
        "✅ Proper StyleGAN architecture with mapping network",
        "✅ Style modulation (AdaIN) and noise injection",
        "✅ Training stabilization (gradient penalty, EMA)",
        "✅ Device support (CUDA, MPS, CPU)",
        "✅ Reproducible with deterministic seeding",
        "✅ Modern code with type hints and documentation",
        "✅ Interactive Streamlit demo",
        "✅ Evaluation metrics (FID, KID, Precision/Recall)",
        "✅ Configuration-driven training",
        "✅ Unit tests and CI/CD pipeline"
    ]
    for feature in features:
        print(f"  {feature}")
    print()


def demo_model_creation():
    """Demonstrate model creation and parameter counting."""
    print("🔧 Model Creation Demo:")
    print("-" * 30)
    
    # Create models
    generator = StyleGANGenerator(z_dim=512, img_size=64)
    discriminator = StyleGANDiscriminator(img_size=64)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    print()


def demo_configuration():
    """Demonstrate configuration system."""
    print("⚙️ Configuration System:")
    print("-" * 30)
    
    # Load default config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Default configuration loaded:")
    print(f"  Dataset: {config['data']['dataset']}")
    print(f"  Image size: {config['model']['img_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print()


def demo_device_detection():
    """Demonstrate device detection."""
    print("💻 Device Detection:")
    print("-" * 30)
    
    device = get_device()
    print(f"Detected device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    else:
        print("Using CPU")
    print()


def demo_data_loading():
    """Demonstrate data loading."""
    print("📊 Data Loading Demo:")
    print("-" * 30)
    
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_name="cifar10",
            batch_size=32,
            img_size=64
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Get a sample batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Sample batch shape: {images.shape}")
            print(f"Sample labels: {labels[:5].tolist()}")
            break
            
    except Exception as e:
        print(f"Data loading demo failed: {e}")
    print()


def demo_sample_generation():
    """Demonstrate sample generation."""
    print("🎨 Sample Generation Demo:")
    print("-" * 30)
    
    try:
        # Set seed for reproducibility
        set_seed(42)
        device = get_device()
        
        # Create generator
        generator = StyleGANGenerator(z_dim=512, img_size=64)
        generator = generator.to(device)
        generator.eval()
        
        # Generate samples
        with torch.no_grad():
            z = torch.randn(4, 512, device=device)
            samples = generator(z)
        
        print(f"Generated samples shape: {samples.shape}")
        print(f"Sample value range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Save samples
        save_samples(generator, epoch=0, num_samples=16, device=device)
        print("Sample images saved to assets/samples/")
        
    except Exception as e:
        print(f"Sample generation demo failed: {e}")
    print()


def print_usage_instructions():
    """Print usage instructions."""
    print("📖 Usage Instructions:")
    print("-" * 30)
    
    instructions = [
        "1. Install dependencies:",
        "   pip install -r requirements.txt",
        "",
        "2. Train StyleGAN:",
        "   python scripts/train.py --config configs/default.yaml",
        "",
        "3. Generate samples:",
        "   python scripts/sample.py --checkpoint assets/checkpoints/generator.pth",
        "",
        "4. Evaluate model:",
        "   python scripts/evaluate.py --checkpoint assets/checkpoints/generator.pth",
        "",
        "5. Launch interactive demo:",
        "   streamlit run demo/app.py",
        "",
        "6. Run tests:",
        "   pytest tests/",
        "",
        "7. Format code:",
        "   black src/ tests/ scripts/",
        "   ruff check src/ tests/ scripts/"
    ]
    
    for instruction in instructions:
        print(instruction)
    print()


def main():
    """Main demonstration function."""
    print_project_info()
    demo_model_creation()
    demo_configuration()
    demo_device_detection()
    demo_data_loading()
    demo_sample_generation()
    print_usage_instructions()
    
    print("=" * 60)
    print("✅ StyleGAN project is ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
