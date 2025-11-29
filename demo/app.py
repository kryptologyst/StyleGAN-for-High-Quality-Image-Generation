"""
Streamlit demo application for StyleGAN.

This app provides an interactive interface for generating samples
from a trained StyleGAN model with various controls.
"""

import streamlit as st
import torch
import torchvision
import numpy as np
from PIL import Image
import yaml
from pathlib import Path

from src.models.generator import StyleGANGenerator
from src.utils.device import get_device


@st.cache_resource
def load_model(config_path: str, checkpoint_path: str):
    """Load the StyleGAN model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = StyleGANGenerator(
        z_dim=config['model']['z_dim'],
        w_dim=config['model']['w_dim'],
        img_channels=config['model']['img_channels'],
        img_size=config['model']['img_size'],
        mapping_layers=config['model']['mapping_layers']
    )
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator = generator.to(device)
    generator.eval()
    
    return generator, config, device


def generate_samples(generator, num_samples, seed, device):
    """Generate samples from the generator."""
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        
        z = torch.randn(num_samples, generator.z_dim, device=device)
        fake_images = generator(z).cpu()
        
        # Convert to PIL images
        images = []
        for i in range(fake_images.shape[0]):
            img = fake_images[i]
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            # Convert to PIL
            img = torchvision.transforms.ToPILImage()(img)
            images.append(img)
        
        return images


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="StyleGAN Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 StyleGAN Demo")
    st.markdown("Generate high-quality images using StyleGAN")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model selection
    config_path = st.sidebar.selectbox(
        "Configuration",
        ["configs/default.yaml"],
        help="Select the model configuration"
    )
    
    checkpoint_path = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pth', 'pt'],
        help="Upload a trained StyleGAN checkpoint"
    )
    
    if checkpoint_path is None:
        st.warning("Please upload a model checkpoint to generate samples.")
        st.stop()
    
    # Save uploaded file temporarily
    temp_path = f"temp_checkpoint_{checkpoint_path.name}"
    with open(temp_path, "wb") as f:
        f.write(checkpoint_path.getbuffer())
    
    try:
        # Load model
        generator, config, device = load_model(config_path, temp_path)
        
        # Generation controls
        st.sidebar.subheader("Generation Settings")
        
        num_samples = st.sidebar.slider(
            "Number of samples",
            min_value=1,
            max_value=64,
            value=16,
            help="Number of images to generate"
        )
        
        seed = st.sidebar.number_input(
            "Random seed",
            min_value=0,
            max_value=2**32-1,
            value=42,
            help="Seed for reproducible generation"
        )
        
        # Generate button
        if st.sidebar.button("Generate Samples", type="primary"):
            with st.spinner("Generating samples..."):
                images = generate_samples(generator, num_samples, seed, device)
            
            # Display images
            st.subheader(f"Generated {len(images)} samples")
            
            # Create columns for grid display
            cols = st.columns(4)
            for i, img in enumerate(images):
                with cols[i % 4]:
                    st.image(img, caption=f"Sample {i+1}")
            
            # Download option
            if st.button("Download All Images"):
                # Create a zip file with all images
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i, img in enumerate(images):
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        zip_file.writestr(f"sample_{i+1:03d}.png", img_buffer.getvalue())
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="stylegan_samples.zip",
                    mime="application/zip"
                )
        
        # Model info
        st.sidebar.subheader("Model Info")
        st.sidebar.info(f"""
        **Configuration:** {config_path}
        **Device:** {device}
        **Image Size:** {config['model']['img_size']}x{config['model']['img_size']}
        **Latent Dim:** {config['model']['z_dim']}
        """)
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    finally:
        # Clean up temporary file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


if __name__ == "__main__":
    main()
