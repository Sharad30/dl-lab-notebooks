# TinyImageNet Autoencoder Training System

A comprehensive PyTorch-based system for training autoencoders on the TinyImageNet dataset. This package provides multiple autoencoder architectures, training utilities, and visualization tools.

## Features

- **Multiple Architectures**: Basic, Convolutional, Variational (VAE), and ResNet-style autoencoders
- **Automatic Dataset Handling**: Downloads and preprocesses TinyImageNet dataset
- **Comprehensive Training**: Progress tracking, checkpointing, and tensorboard logging
- **Rich Visualizations**: Reconstruction comparisons, latent space visualization, interpolations
- **Easy to Use**: Simple API designed for notebook usage

## Quick Start

### Basic Usage in Notebook

```python
# Import the main functions
from notebooks.autoencoder import train_basic_autoencoder, analyze_trained_model

# Train a convolutional autoencoder
model, trainer, train_loader, val_loader = train_basic_autoencoder(
    epochs=20,
    batch_size=64
)

# Analyze and visualize results
analyze_trained_model(model, trainer, train_loader, val_loader)
```

### Training Different Models

```python
from notebooks.autoencoder import train_vae, train_resnet_autoencoder

# Train a Variational Autoencoder
vae_model, vae_trainer, train_loader, val_loader = train_vae(epochs=30)

# Train a ResNet-style autoencoder
resnet_model, resnet_trainer, train_loader, val_loader = train_resnet_autoencoder(epochs=50)
```

### Quick Demo

```python
from notebooks.autoencoder import quick_demo

# Run a quick 5-epoch demo to test everything works
model, trainer, train_loader, val_loader = quick_demo()
```

## Detailed Usage Examples

### 1. Custom Training Configuration

```python
from notebooks.autoencoder import get_tinyimagenet_dataloaders, get_model, create_trainer

# Load dataset with custom settings
train_loader, val_loader = get_tinyimagenet_dataloaders(
    root_dir='./my_data',
    batch_size=32,
    image_size=64,
    num_workers=8
)

# Create a custom model
model = get_model('conv', latent_dim=512, input_channels=3)

# Create trainer
trainer = create_trainer(model, model_type='standard')

# Train with custom parameters
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-4,
    optimizer_type='adam',
    scheduler_type='cosine',
    save_dir='my_checkpoints'
)
```

### 2. Loading and Using Pretrained Models

```python
from notebooks.autoencoder import load_pretrained_model, visualize_reconstructions

# Load a pretrained model
model, checkpoint_info = load_pretrained_model(
    'checkpoints/best_model.pth',
    model_type='conv',
    latent_dim=512
)

# Use the model for visualization
visualize_reconstructions(model, val_loader, device='cuda', num_samples=8)
```

### 3. Comprehensive Model Analysis

```python
from notebooks.autoencoder.utils import (
    visualize_latent_space,
    interpolate_latent_space,
    compute_reconstruction_error,
    analyze_latent_dimensions
)

# Visualize latent space with t-SNE
visualize_latent_space(model, val_loader, device='cuda', method='tsne')

# Interpolate between images
interpolate_latent_space(model, val_loader, device='cuda')

# Compute reconstruction statistics
error_stats = compute_reconstruction_error(model, val_loader, device='cuda')
print(f"Mean reconstruction error: {error_stats['mean_error']:.4f}")

# Analyze latent dimensions
latent_analysis = analyze_latent_dimensions(model, val_loader, device='cuda')
print(f"Effective dimensions: {latent_analysis['effective_dims']}")
```

### 4. VAE-Specific Features

```python
from notebooks.autoencoder import train_vae
from notebooks.autoencoder.utils import generate_random_samples

# Train VAE
vae_model, vae_trainer, train_loader, val_loader = train_vae()

# Generate new samples from random latent codes
generate_random_samples(vae_model, device='cuda', num_samples=16, model_type='vae')
```

### 5. Model Comparison

```python
from notebooks.autoencoder import compare_models

# Compare different architectures
compare_models()  # This will train and compare basic, conv, and VAE models
```

## Available Models

### 1. BasicAutoencoder
- Fully connected layers
- Good for understanding basic concepts
- Smaller memory footprint

```python
model = get_model('basic', input_dim=3*64*64, latent_dim=256)
```

### 2. ConvAutoencoder
- Convolutional encoder/decoder
- Better for image data
- Recommended for most use cases

```python
model = get_model('conv', latent_dim=512)
```

### 3. VariationalAutoencoder (VAE)
- Probabilistic latent space
- Can generate new samples
- Good for generative tasks

```python
model = get_model('vae', latent_dim=256)
```

### 4. ResNetAutoencoder
- Residual connections
- Deeper architecture
- Best reconstruction quality

```python
model = get_model('resnet', latent_dim=512, num_residual_blocks=2)
```

## Configuration Recommendations

### For Beginners
```python
config = {
    'model_type': 'conv',
    'latent_dim': 256,
    'epochs': 20,
    'batch_size': 64,
    'lr': 1e-3
}
```

### For Best Quality
```python
config = {
    'model_type': 'resnet',
    'latent_dim': 512,
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4
}
```

### For Generative Tasks
```python
config = {
    'model_type': 'vae',
    'latent_dim': 256,
    'epochs': 80,
    'batch_size': 64,
    'lr': 1e-3
}
```

## File Structure

```
notebooks/autoencoder/
├── __init__.py              # Package initialization and convenient imports
├── tinyimagenet_dataset.py  # Dataset loading and preprocessing
├── models.py               # Autoencoder architectures
├── trainer.py             # Training system
├── utils.py               # Visualization and analysis utilities
├── example_usage.py       # Comprehensive examples
└── README.md             # This file
```

## Requirements

The following packages are required (already included in the project):
- torch >= 2.7.0
- torchvision >= 0.22.0
- matplotlib >= 3.10.3
- tqdm >= 4.67.1
- scipy >= 1.15.3

Additional packages used in utils.py:
- scikit-learn (for t-SNE and PCA)
- seaborn (for enhanced plotting)

## Training Tips

1. **Start Small**: Use the `quick_demo()` function to test everything works
2. **Monitor Training**: Check the loss plots and reconstruction visualizations
3. **Adjust Learning Rate**: Start with 1e-3, reduce if training is unstable
4. **Latent Dimension**: 256-512 works well for TinyImageNet
5. **Batch Size**: Use 64 for conv models, 32 for ResNet models
6. **Checkpoints**: Models are automatically saved during training

## Common Issues and Solutions

### Out of Memory
- Reduce batch size
- Use BasicAutoencoder instead of ResNet
- Reduce latent dimension

### Poor Reconstruction Quality
- Increase latent dimension
- Train for more epochs
- Try ResNetAutoencoder
- Check learning rate (try 1e-4)

### VAE Generating Blurry Images
- Adjust beta parameter in trainer.py
- Increase latent dimension
- Train for more epochs

## Getting Help

Use the built-in recommendations:
```python
from notebooks.autoencoder import get_recommendations
get_recommendations()
```

This will print detailed recommendations for different use cases. 