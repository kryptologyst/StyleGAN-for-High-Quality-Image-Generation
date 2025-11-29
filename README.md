# StyleGAN for High-Quality Image Generation

A production-ready implementation of StyleGAN (Style Generative Adversarial Network) for generating high-quality, photorealistic images with fine-grained style control.

## Features

- **Proper StyleGAN Architecture**: Implements the actual StyleGAN paper with mapping network, style modulation (AdaIN), noise injection, and synthesis blocks
- **Training Stabilization**: Includes gradient penalty (WGAN-GP), exponential moving average (EMA), equalized learning rates, and minibatch standard deviation
- **Device Support**: Automatic detection and support for CUDA, MPS (Apple Silicon), and CPU
- **Reproducible**: Deterministic seeding for all random operations
- **Modern Stack**: Clean, typed code with proper documentation and configuration management
- **Interactive Demo**: Streamlit web application for easy sample generation
- **Evaluation Ready**: Built-in support for FID, KID, and other evaluation metrics

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/StyleGAN-for-High-Quality-Image-Generation.git
cd StyleGAN-for-High-Quality-Image-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train StyleGAN on CIFAR-10:
```bash
python scripts/train.py --config configs/default.yaml
```

Train with custom parameters:
```bash
python scripts/train.py --dataset cifar10 --epochs 200 --batch-size 128 --lr 0.0001
```

### Sampling

Generate samples from a trained model:
```bash
python scripts/sample.py --checkpoint assets/checkpoints/generator.pth --num-samples 64 --seed 42
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
stylegan-high-quality-generation/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── generator.py   # StyleGAN Generator
│   │   └── discriminator.py # StyleGAN Discriminator
│   ├── training/          # Training utilities
│   │   └── trainer.py     # Training functions
│   ├── data/              # Data handling
│   │   └── dataloader.py  # Data loaders
│   └── utils/              # Utilities
│       └── device.py      # Device detection
├── configs/               # Configuration files
│   └── default.yaml       # Default configuration
├── scripts/               # Training and sampling scripts
│   ├── train.py          # Training script
│   └── sample.py         # Sampling script
├── demo/                  # Demo application
│   └── app.py            # Streamlit demo
├── assets/               # Generated assets
│   ├── samples/          # Generated samples
│   └── checkpoints/      # Model checkpoints
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Configuration

The training process is controlled by YAML configuration files. Key parameters:

### Model Parameters
- `z_dim`: Latent vector dimension (default: 512)
- `w_dim`: Style vector dimension (default: 512)
- `img_size`: Output image size (default: 64)
- `img_channels`: Number of image channels (default: 3)

### Training Parameters
- `batch_size`: Batch size (default: 64)
- `num_epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 0.0002)
- `lambda_gp`: Gradient penalty weight (default: 10.0)

### Data Parameters
- `dataset`: Dataset name (cifar10, mnist)
- `num_workers`: Number of data loading workers

## Architecture Details

### Generator
The StyleGAN generator consists of:

1. **Mapping Network**: Transforms random noise z to style vector w
2. **Constant Input**: Learnable constant tensor as starting point
3. **Synthesis Network**: Progressive upsampling blocks with style modulation
4. **Style Modulation**: Adaptive instance normalization (AdaIN) for style control
5. **Noise Injection**: Adds stochastic variation at each layer

### Discriminator
The discriminator features:

1. **Progressive Downsampling**: Reduces spatial resolution while increasing channels
2. **Minibatch Standard Deviation**: Improves training stability
3. **Skip Connections**: Helps with gradient flow
4. **Equalized Learning Rates**: Stabilizes training

## Training Stabilization Techniques

- **Gradient Penalty**: Implements WGAN-GP for stable training
- **Exponential Moving Average**: Maintains stable generator parameters
- **Equalized Learning Rates**: Scales weights by 1/sqrt(fan_in)
- **Minibatch Standard Deviation**: Reduces mode collapse
- **Deterministic Operations**: Ensures reproducibility

## Evaluation Metrics

The implementation supports standard GAN evaluation metrics:

- **FID (Fréchet Inception Distance)**: Measures quality and diversity
- **KID (Kernel Inception Distance)**: Unbiased alternative to FID
- **Precision/Recall**: Measures quality and coverage
- **LPIPS**: Perceptual similarity metric

## Datasets

Currently supported datasets:
- **CIFAR-10**: 32x32 color images (10 classes)
- **MNIST**: 28x28 grayscale images (10 digits)

## Usage Examples

### Basic Training
```python
from src.models.generator import StyleGANGenerator
from src.models.discriminator import StyleGANDiscriminator
from src.training.trainer import train_stylegan
from src.data.dataloader import create_data_loaders

# Create models
generator = StyleGANGenerator(z_dim=512, img_size=64)
discriminator = StyleGANDiscriminator(img_size=64)

# Create data loaders
train_loader, val_loader = create_data_loaders("cifar10", batch_size=64)

# Train
train_stylegan(generator, discriminator, train_loader, num_epochs=100)
```

### Custom Configuration
```python
import yaml

# Load custom config
with open("configs/custom.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use config parameters
generator = StyleGANGenerator(
    z_dim=config["model"]["z_dim"],
    img_size=config["model"]["img_size"]
)
```

## Performance Tips

1. **Batch Size**: Use larger batch sizes (128+) for better stability
2. **Learning Rate**: Start with 0.0002, adjust based on training dynamics
3. **Gradient Penalty**: Increase lambda_gp if discriminator becomes too strong
4. **EMA Decay**: Higher values (0.999) for more stable samples
5. **Device**: Use CUDA for significantly faster training

## Troubleshooting

### Common Issues

1. **Mode Collapse**: Increase gradient penalty weight or reduce learning rate
2. **Training Instability**: Ensure proper weight initialization and equalized learning rates
3. **Memory Issues**: Reduce batch size or image resolution
4. **Poor Quality**: Check data preprocessing and normalization

### Debug Mode
Enable debug logging:
```bash
python scripts/train.py --config configs/debug.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original StyleGAN paper by Karras et al.
- PyTorch team for the excellent framework
- Streamlit team for the demo framework
# StyleGAN-for-High-Quality-Image-Generation
