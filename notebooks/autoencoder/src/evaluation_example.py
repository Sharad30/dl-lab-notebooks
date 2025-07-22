"""
Example script demonstrating comprehensive autoencoder evaluation.

This script shows how to use the new evaluation metrics beyond MSE loss,
including SSIM, LPIPS, FID, and latent space analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os

# Import your autoencoder models and trainer
from models import StandardAutoencoder, VariationalAutoencoder  # Adjust imports as needed
from trainer import AutoencoderTrainer


def setup_data_loaders(data_dir='./data', batch_size=32, image_size=32):
    """Setup data loaders for evaluation."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def comprehensive_evaluation_example():
    """Example of comprehensive autoencoder evaluation."""
    
    print("=== Comprehensive Autoencoder Evaluation Example ===\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = setup_data_loaders()
    
    # Create model (adjust architecture as needed)
    model = StandardAutoencoder(
        input_channels=3,
        latent_dim=128,
        image_size=32
    )
    
    # Create trainer with comprehensive evaluation capabilities
    trainer = AutoencoderTrainer(
        model=model,
        device=device,
        model_type='standard'  # or 'vae' for variational autoencoder
    )
    
    print("Available evaluation metrics:")
    print("- MSE (Mean Squared Error)")
    print("- PSNR (Peak Signal-to-Noise Ratio)")
    print("- SSIM (Structural Similarity Index)")
    print("- LPIPS (Learned Perceptual Image Patch Similarity)")
    print("- FID (Fréchet Inception Distance)")
    print("- Latent Space Analysis")
    
    # Load pre-trained model (if available)
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"\nLoading pre-trained model from {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print(f"\nNo pre-trained model found at {checkpoint_path}")
        print("Training a simple model for demonstration...")
        
        # Quick training for demonstration
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,  # Small number for demo
            lr=1e-3,
            save_dir='checkpoints'
        )
    
    # Perform comprehensive evaluation
    print("\n" + "="*60)
    print("PERFORMING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Evaluate on validation and test sets
    results = trainer.post_training_evaluation(
        val_loader=val_loader,
        test_loader=test_loader,  # This is the unseen data!
        save_dir='evaluation_results',
        num_samples=500  # Evaluate on 500 samples for speed
    )
    
    # Additional analysis: Compare training vs validation vs test performance
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS")
    print("="*60)
    
    if results['test_metrics'] is not None:
        val_mse = results['validation_metrics']['reconstruction_metrics']['mse_mean']
        test_mse = results['test_metrics']['reconstruction_metrics']['mse_mean']
        
        print(f"Validation MSE: {val_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Generalization gap: {((test_mse - val_mse) / val_mse * 100):.2f}%")
        
        if (test_mse - val_mse) / val_mse > 0.1:  # 10% degradation threshold
            print("⚠️  Warning: Significant performance degradation on unseen data!")
            print("   Consider: regularization, more diverse training data, or architecture changes")
        else:
            print("✅ Good generalization to unseen data!")
    
    # Demonstrate individual metric computation
    print("\n" + "="*60)
    print("INDIVIDUAL METRIC COMPUTATION EXAMPLE")
    print("="*60)
    
    # Get a single batch for demonstration
    data_iter = iter(val_loader)
    batch_data = next(data_iter)
    
    # Extract images
    images = trainer._extract_images_from_batch(batch_data)
    
    # Get reconstructions
    trainer.model.eval()
    with torch.no_grad():
        if trainer.model_type == 'vae':
            reconstructions, _, _ = trainer.model(images)
        else:
            reconstructions = trainer.model(images)
    
    # Compute individual metrics
    print(f"Batch size: {images.size(0)}")
    
    # MSE
    mse = torch.nn.functional.mse_loss(reconstructions, images).item()
    print(f"Batch MSE: {mse:.6f}")
    
    # SSIM (for first image in batch)
    ssim = trainer.compute_ssim(images[0:1], reconstructions[0:1])
    print(f"SSIM (first image): {ssim:.4f}")
    
    # LPIPS (for batch)
    lpips = trainer.compute_lpips(images, reconstructions)
    print(f"LPIPS (batch average): {lpips:.4f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Check the 'evaluation_results' directory for:")
    print("- comprehensive_evaluation.json: Detailed numerical results")
    print("- metrics_comparison.png: Visual comparison plots")
    
    return results


def metric_interpretation_guide():
    """Print guide for interpreting the different metrics."""
    
    print("\n" + "="*60)
    print("METRIC INTERPRETATION GUIDE")
    print("="*60)
    
    print("\n1. RECONSTRUCTION METRICS:")
    print("   • MSE (Mean Squared Error): Lower is better")
    print("     - Measures pixel-wise differences")
    print("     - Range: [0, ∞], typical: [0.001, 0.1]")
    
    print("   • PSNR (Peak Signal-to-Noise Ratio): Higher is better")
    print("     - Measures reconstruction quality in dB")
    print("     - Range: [0, ∞], good: >20dB, excellent: >30dB")
    
    print("\n2. PERCEPTUAL METRICS:")
    print("   • SSIM (Structural Similarity): Higher is better")
    print("     - Measures structural similarity")
    print("     - Range: [-1, 1], good: >0.8, excellent: >0.9")
    
    print("   • LPIPS (Learned Perceptual Similarity): Lower is better")
    print("     - Measures perceptual similarity using deep features")
    print("     - Range: [0, ∞], good: <0.1, excellent: <0.05")
    
    print("\n3. DISTRIBUTIONAL METRICS:")
    print("   • FID (Fréchet Inception Distance): Lower is better")
    print("     - Measures distribution similarity")
    print("     - Range: [0, ∞], good: <50, excellent: <10")
    
    print("\n4. LATENT SPACE ANALYSIS:")
    print("   • Active Dimensions: Higher often better")
    print("     - Number of dimensions with significant variance")
    print("     - Indicates representation richness")
    
    print("   • Total Variance: Context-dependent")
    print("     - Total variance in latent space")
    print("     - Too low: under-utilization, too high: noise")


if __name__ == "__main__":
    # Run the comprehensive evaluation example
    results = comprehensive_evaluation_example()
    
    # Print interpretation guide
    metric_interpretation_guide()
    
    print(f"\nFinal Results Summary:")
    if results['test_metrics']:
        test_metrics = results['test_metrics']
        print(f"Test MSE: {test_metrics['reconstruction_metrics']['mse_mean']:.6f}")
        print(f"Test SSIM: {test_metrics['perceptual_metrics']['ssim_mean']:.4f}")
        print(f"Test LPIPS: {test_metrics['perceptual_metrics']['lpips_mean']:.4f}")
        print(f"Test FID: {test_metrics['distributional_metrics']['fid_score']:.2f}") 