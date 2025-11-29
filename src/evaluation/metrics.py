"""
Evaluation metrics for StyleGAN.

This module contains implementations of FID, KID, Precision/Recall,
and other evaluation metrics for generative models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import List, Tuple, Optional
from scipy import linalg


class InceptionV3(nn.Module):
    """InceptionV3 model for feature extraction."""
    
    def __init__(self, normalize_input: bool = True) -> None:
        super().__init__()
        self.normalize_input = normalize_input
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(inception.children())[:-1])
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        if self.normalize_input:
            x = (x + 1) / 2  # Normalize from [-1, 1] to [0, 1]
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        return features


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray
) -> float:
    """Calculate Fréchet Inception Distance (FID).
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid


def calculate_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    kernel: str = 'rbf',
    gamma: Optional[float] = None
) -> float:
    """Calculate Kernel Inception Distance (KID).
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        kernel: Kernel type for MMD calculation
        gamma: Kernel parameter
        
    Returns:
        KID score (lower is better)
    """
    from sklearn.metrics import pairwise_kernels
    
    # Combine features
    all_features = np.vstack([real_features, fake_features])
    
    # Create labels
    n_real = len(real_features)
    n_fake = len(fake_features)
    labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
    
    # Calculate kernel matrix
    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / all_features.shape[1]
        K = pairwise_kernels(all_features, metric='rbf', gamma=gamma)
    else:
        K = pairwise_kernels(all_features, metric=kernel)
    
    # Calculate MMD
    K_real = K[:n_real, :n_real]
    K_fake = K[n_real:, n_real:]
    K_cross = K[:n_real, n_real:]
    
    mmd = (K_real.mean() + K_fake.mean() - 2 * K_cross.mean())
    
    return mmd


def calculate_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 5
) -> Tuple[float, float]:
    """Calculate Precision and Recall for generative models.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (precision, recall)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit nearest neighbors on real data
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_real.fit(real_features)
    
    # Fit nearest neighbors on fake data
    nn_fake = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_fake.fit(fake_features)
    
    # Calculate precision: how many fake samples have real neighbors
    distances_fake, _ = nn_real.kneighbors(fake_features)
    precision = np.mean(distances_fake[:, -1] < np.percentile(distances_fake, 50))
    
    # Calculate recall: how many real samples have fake neighbors
    distances_real, _ = nn_fake.kneighbors(real_features)
    recall = np.mean(distances_real[:, -1] < np.percentile(distances_real, 50))
    
    return precision, recall


def evaluate_generator(
    generator: nn.Module,
    real_loader,
    device: torch.device,
    num_samples: int = 10000,
    batch_size: int = 100
) -> dict:
    """Evaluate generator with multiple metrics.
    
    Args:
        generator: Trained generator model
        real_loader: DataLoader for real images
        device: Device to run evaluation on
        num_samples: Number of samples to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    generator.eval()
    inception = InceptionV3().to(device)
    
    # Extract real features
    real_features = []
    with torch.no_grad():
        for batch_idx, (real_images, _) in enumerate(real_loader):
            if batch_idx * batch_size >= num_samples:
                break
            
            real_images = real_images.to(device)
            features = inception(real_images)
            real_features.append(features.cpu().numpy())
    
    real_features = np.vstack(real_features)[:num_samples]
    
    # Generate fake features
    fake_features = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            z = torch.randn(current_batch_size, generator.z_dim, device=device)
            fake_images = generator(z)
            features = inception(fake_images)
            fake_features.append(features.cpu().numpy())
    
    fake_features = np.vstack(fake_features)
    
    # Calculate metrics
    fid_score = calculate_fid(real_features, fake_features)
    kid_score = calculate_kid(real_features, fake_features)
    precision, recall = calculate_precision_recall(real_features, fake_features)
    
    return {
        'fid': fid_score,
        'kid': kid_score,
        'precision': precision,
        'recall': recall
    }
