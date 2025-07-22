"""
Complete example of training an autoencoder on TinyImageNet dataset.

This script demonstrates how to use all the autoencoder components together:
- Loading TinyImageNet dataset
- Creating different autoencoder models
- Training with proper monitoring
- Visualizing results

You can import functions from this file in your notebook or run it as a script.
"""

import torch
import torch.nn as nn
from tinyimagenet_dataset import get_tinyimagenet_dataloaders
from models import get_model, count_parameters
from trainer import create_trainer
from utils import (
    visualize_reconstructions, 
    visualize_latent_space,
    generate_random_samples,
    interpolate_latent_space,
    compute_reconstruction_error,
    plot_loss_components
)


def train_basic_autoencoder(data_root='./data', epochs=50, batch_size=64):
    """
    Train a basic convolutional autoencoder.
    
    Args:
        data_root: Directory to store/load the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: (model, trainer, train_loader, val_loader)
    """
    print("=== Training Basic Convolutional Autoencoder ===")
    
    # Load dataset
    print("Loading TinyImageNet dataset...")
    train_loader, val_loader = get_tinyimagenet_dataloaders(
        root_dir=data_root, 
        batch_size=batch_size,
        download=True
    )
    print("Dataset loaded successfully")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model('conv', latent_dim=512)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = create_trainer(model, model_type='standard')
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        optimizer_type='adam',
        scheduler_type='step',
        save_dir='checkpoints/conv_autoencoder'
    )
    
    return model, trainer, train_loader, val_loader


def train_vae(data_root='./data', epochs=50, batch_size=64):
    """
    Train a Variational Autoencoder.
    
    Args:
        data_root: Directory to store/load the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: (model, trainer, train_loader, val_loader)
    """
    print("=== Training Variational Autoencoder ===")
    
    # Load dataset
    print("Loading TinyImageNet dataset...")
    train_loader, val_loader = get_tinyimagenet_dataloaders(
        root_dir=data_root, 
        batch_size=batch_size,
        download=True
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model('vae', latent_dim=256)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = create_trainer(model, model_type='vae')
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        optimizer_type='adam',
        scheduler_type='cosine',
        save_dir='checkpoints/vae'
    )
    
    return model, trainer, train_loader, val_loader


def train_resnet_autoencoder(data_root='./data', epochs=50, batch_size=32):
    """
    Train a ResNet-style autoencoder.
    
    Args:
        data_root: Directory to store/load the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: (model, trainer, train_loader, val_loader)
    """
    print("=== Training ResNet Autoencoder ===")
    
    # Load dataset (smaller batch size for ResNet due to memory)
    print("Loading TinyImageNet dataset...")
    train_loader, val_loader = get_tinyimagenet_dataloaders(
        root_dir=data_root, 
        batch_size=batch_size,
        download=True
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model('resnet', latent_dim=512, num_residual_blocks=2)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = create_trainer(model, model_type='standard')
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        optimizer_type='adam',
        scheduler_type='step',
        save_dir='checkpoints/resnet_autoencoder'
    )
    
    return model, trainer, train_loader, val_loader


def analyze_trained_model(model, trainer, train_loader, val_loader, model_type='standard'):
    """
    Comprehensive analysis of a trained autoencoder model.
    
    Args:
        model: Trained autoencoder model
        trainer: Trainer used for training
        train_loader: Training data loader
        val_loader: Validation data loader
        model_type: Type of model ('standard', 'vae')
    """
    print("=== Analyzing Trained Model ===")
    device = next(model.parameters()).device
    
    # Plot training history
    print("Plotting training history...")
    plot_loss_components(trainer)
    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    visualize_reconstructions(
        model, val_loader, device, 
        num_samples=8, model_type=model_type
    )
    
    # Visualize latent space
    print("Visualizing latent space...")
    visualize_latent_space(
        model, val_loader, device, 
        method='tsne', num_samples=1000
    )
    
    # Interpolate in latent space
    print("Interpolating in latent space...")
    interpolate_latent_space(
        model, val_loader, device, model_type=model_type
    )
    
    # Generate random samples (for VAE)
    if model_type == 'vae':
        print("Generating random samples...")
        generate_random_samples(
            model, device, num_samples=16, model_type=model_type
        )
    
    # Compute reconstruction error statistics
    print("Computing reconstruction error statistics...")
    error_stats = compute_reconstruction_error(
        model, val_loader, device, model_type=model_type
    )
    
    print("Reconstruction Error Statistics:")
    for key, value in error_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def quick_demo():
    """
    Quick demonstration with a small model and few epochs.
    Perfect for testing the setup.
    """
    print("=== Quick Demo ===")
    
    # Load small subset for demo
    train_loader, val_loader = get_tinyimagenet_dataloaders(
        root_dir='../visual_search_engine/data', 
        batch_size=32,
        download=True
    )
    
    # Create small model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('conv', latent_dim=128)  # Smaller latent dimension
    
    # Create trainer
    trainer = create_trainer(model, model_type='standard')
    
    # Train for just a few epochs
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Very few epochs for demo
        lr=1e-3,
        save_dir='checkpoints/demo'
    )
    
    # Quick analysis
    print("Demo complete! Visualizing results...")
    visualize_reconstructions(model, val_loader, device, num_samples=4)
    
    return model, trainer, train_loader, val_loader


def compare_models():
    """
    Compare different autoencoder architectures.
    """
    print("=== Comparing Different Models ===")
    
    # Load dataset once
    train_loader, val_loader = get_tinyimagenet_dataloaders(
        root_dir='./data', 
        batch_size=32,
        download=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    # Train different models
    model_configs = [
        ('basic', {'input_dim': 3*64*64, 'latent_dim': 256}),
        ('conv', {'latent_dim': 256}),
        ('vae', {'latent_dim': 256}),
    ]
    
    for model_name, config in model_configs:
        print(f"\nTraining {model_name} autoencoder...")
        
        model = get_model(model_name, **config)
        model_type = 'vae' if model_name == 'vae' else 'standard'
        trainer = create_trainer(model, model_type=model_type)
        
        # Train for limited epochs
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            lr=1e-3,
            save_dir=f'checkpoints/comparison/{model_name}'
        )
        
        models[model_name] = {
            'model': model,
            'trainer': trainer,
            'type': model_type
        }
    
    # Compare results
    print("\n=== Model Comparison Results ===")
    for name, info in models.items():
        trainer = info['trainer']
        final_train_loss = trainer.train_losses[-1]
        final_val_loss = trainer.val_losses[-1]
        params = count_parameters(info['model'])
        
        print(f"\n{name.upper()} Autoencoder:")
        print(f"  Parameters: {params:,}")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        
        # Visualize reconstructions for each model
        print(f"  Visualizing {name} reconstructions...")
        visualize_reconstructions(
            info['model'], val_loader, device, 
            num_samples=4, model_type=info['type']
        )


# Utility functions for notebook usage
def load_pretrained_model(checkpoint_path, model_type='conv', **model_kwargs):
    """
    Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_type: Type of model to create
        **model_kwargs: Additional arguments for model creation
        
    Returns:
        tuple: (model, checkpoint_info)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(model_type, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, checkpoint


def get_recommendations():
    """
    Print recommendations for different use cases.
    """
    print("=== Model Recommendations ===")
    print()
    print("1. **For beginners/quick experiments:**")
    print("   - Use ConvAutoencoder with latent_dim=256")
    print("   - Train for 20-50 epochs")
    print("   - Batch size: 64")
    print()
    print("2. **For best reconstruction quality:**")
    print("   - Use ResNetAutoencoder with residual blocks")
    print("   - Larger latent dimension (512+)")
    print("   - Train for 100+ epochs")
    print()
    print("3. **For generative capabilities:**")
    print("   - Use VariationalAutoencoder (VAE)")
    print("   - Moderate latent dimension (128-256)")
    print("   - Tune beta parameter for KL loss")
    print()
    print("4. **For memory-constrained environments:**")
    print("   - Use BasicAutoencoder")
    print("   - Smaller batch sizes (16-32)")
    print("   - Reduce latent dimension")
    print()
    print("5. **Training tips:**")
    print("   - Start with lower learning rates (1e-4)")
    print("   - Use learning rate scheduling")
    print("   - Monitor both reconstruction and validation loss")
    print("   - Save checkpoints regularly")


if __name__ == "__main__":
    # Run quick demo if executed as script
    print("Running quick demo...")
    quick_demo()
    print("\nDemo completed! You can now use these functions in your notebook.")
    print("\nTo get started:")
    print("1. Import the functions you need")
    print("2. Call train_basic_autoencoder() for a full training run")
    print("3. Use analyze_trained_model() to visualize results")
    print("4. Call get_recommendations() for usage tips") 