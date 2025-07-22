import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean


def visualize_reconstructions(model, dataloader, device, num_samples=8, 
                            save_path=None, model_type='standard'):
    """
    Visualize original vs reconstructed images.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader for visualization
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        model_type: Type of model ('standard', 'vae')
    """
    model.eval()
    
    # Get a batch of data
    with torch.no_grad():
        data_iter = iter(dataloader)
        batch_data = next(data_iter)
        
        # Extract images from the batch (handle both old and new dataset formats)
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            images = batch_data[0][:num_samples].to(device)  # Get images (first element)
        else:
            images = batch_data[:num_samples].to(device)  # Fallback for old format
        
        # Generate reconstructions
        if model_type == 'vae':
            reconstructions, _, _ = model(images)
        else:
            reconstructions = model(images)
    
    # Denormalize for visualization
    images_denorm = denormalize_tensor(images.cpu())
    reconstructions_denorm = denormalize_tensor(reconstructions.cpu())
    
    # Clamp values to [0, 1]
    images_denorm = torch.clamp(images_denorm, 0, 1)
    reconstructions_denorm = torch.clamp(reconstructions_denorm, 0, 1)
    
    # Create comparison grid
    comparison = torch.cat([images_denorm, reconstructions_denorm], dim=0)
    grid = make_grid(comparison, nrow=num_samples, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Original (top) vs Reconstructed (bottom)', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def visualize_latent_space(model, dataloader, device, method='tsne', 
                          num_samples=1000, save_path=None):
    """
    Visualize the latent space of the autoencoder.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader
        device: Device to run inference on
        method: Dimensionality reduction method ('tsne', 'pca')
        num_samples: Number of samples to use
        save_path: Path to save the visualization
    """
    model.eval()
    
    latent_vectors = []
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if len(latent_vectors) * dataloader.batch_size >= num_samples:
                break
                
            # Extract images from the batch
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                batch = batch_data[0].to(device)  # Get images (first element)
            else:
                batch = batch_data.to(device)  # Fallback for old format
            
            # Get latent representation
            if hasattr(model, 'encode'):
                if isinstance(model.encode(batch), tuple):  # VAE case
                    latent, _ = model.encode(batch)
                else:
                    latent = model.encode(batch)
            else:
                # For models without separate encode method
                latent = model.encoder(batch)
            
            latent_vectors.append(latent.cpu().numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    
    # Apply dimensionality reduction
    if method == 'tsne':
        if latent_vectors.shape[1] > 50:
            # Apply PCA first if dimensionality is too high
            pca = PCA(n_components=50)
            latent_vectors = pca.fit_transform(latent_vectors)
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(latent_vectors)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_vectors)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)
    plt.title(f'Latent Space Visualization ({method.upper()})', fontsize=16)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def generate_random_samples(model, device, num_samples=16, latent_dim=None, 
                          save_path=None, model_type='vae'):
    """
    Generate random samples from the latent space (works best with VAE).
    
    Args:
        model: Trained model
        device: Device to run inference on
        num_samples: Number of samples to generate
        latent_dim: Latent space dimension
        save_path: Path to save the visualization
        model_type: Type of model
    """
    if model_type != 'vae':
        print("Warning: Random generation works best with VAE models")
    
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution
        if latent_dim is None:
            latent_dim = model.latent_dim
        
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate samples
        if hasattr(model, 'decode'):
            generated = model.decode(z)
        else:
            generated = model.decoder(z)
    
    # Denormalize and clamp
    generated_denorm = denormalize_tensor(generated.cpu())
    generated_denorm = torch.clamp(generated_denorm, 0, 1)
    
    # Create grid
    grid = make_grid(generated_denorm, nrow=4, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Generated Samples from Random Latent Codes', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def interpolate_latent_space(model, dataloader, device, save_path=None, 
                           model_type='standard'):
    """
    Interpolate between two images in latent space.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader
        device: Device to run inference on
        save_path: Path to save the visualization
        model_type: Type of model
    """
    model.eval()
    
    with torch.no_grad():
        # Get two random images
        data_iter = iter(dataloader)
        batch_data = next(data_iter)
        
        # Extract images from the batch
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            images = batch_data[0][:2].to(device)  # Get first 2 images
        else:
            images = batch_data[:2].to(device)  # Fallback for old format
        
        # Encode to latent space
        if hasattr(model, 'encode'):
            if model_type == 'vae':
                latent1, _ = model.encode(images[0:1])
                latent2, _ = model.encode(images[1:2])
            else:
                latent1 = model.encode(images[0:1])
                latent2 = model.encode(images[1:2])
        else:
            latent1 = model.encoder(images[0:1])
            latent2 = model.encoder(images[1:2])
        
        # Create interpolation
        alphas = torch.linspace(0, 1, 8).to(device)
        interpolated_latents = []
        
        for alpha in alphas:
            interpolated = alpha * latent2 + (1 - alpha) * latent1
            interpolated_latents.append(interpolated)
        
        interpolated_latents = torch.cat(interpolated_latents, dim=0)
        
        # Decode interpolated latents
        if hasattr(model, 'decode'):
            interpolated_images = model.decode(interpolated_latents)
        else:
            interpolated_images = model.decoder(interpolated_latents)
    
    # Denormalize and clamp
    interpolated_denorm = denormalize_tensor(interpolated_images.cpu())
    interpolated_denorm = torch.clamp(interpolated_denorm, 0, 1)
    
    # Create grid
    grid = make_grid(interpolated_denorm, nrow=8, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(16, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Latent Space Interpolation', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def compute_reconstruction_error(model, dataloader, device, model_type='standard'):
    """
    Compute reconstruction error statistics.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader
        device: Device to run inference on
        model_type: Type of model
        
    Returns:
        dict: Statistics about reconstruction error
    """
    model.eval()
    
    errors = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Extract images from the batch
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                batch = batch_data[0].to(device)  # Get images (first element)
            else:
                batch = batch_data.to(device)  # Fallback for old format
            
            # Get reconstruction
            if model_type == 'vae':
                reconstruction, _, _ = model(batch)
            else:
                reconstruction = model(batch)
            
            # Compute MSE for each sample
            mse = F.mse_loss(reconstruction, batch, reduction='none')
            mse = mse.view(batch.size(0), -1).mean(dim=1)
            
            errors.extend(mse.cpu().numpy())
            total_samples += batch.size(0)
    
    errors = np.array(errors)
    
    stats = {
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'min_error': errors.min(),
        'max_error': errors.max(),
        'median_error': np.median(errors),
        'total_samples': total_samples
    }
    
    return stats


def plot_loss_components(trainer, save_path=None):
    """
    Plot detailed loss components for VAE training.
    
    Args:
        trainer: AutoencoderTrainer instance
        save_path: Path to save the plot
    """
    if not hasattr(trainer, 'train_losses') or len(trainer.train_losses) == 0:
        print("No training history found")
        return
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot overall losses
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='Training Loss', alpha=0.7)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', alpha=0.7)
    axes[0].set_title('Overall Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot smoothed losses
    if len(trainer.train_losses) > 10:
        window_size = max(1, len(trainer.train_losses) // 20)
        train_smooth = np.convolve(trainer.train_losses, 
                                 np.ones(window_size)/window_size, mode='valid')
        val_smooth = np.convolve(trainer.val_losses, 
                               np.ones(window_size)/window_size, mode='valid')
        epochs_smooth = range(window_size, len(trainer.train_losses) + 1)
        
        axes[1].plot(epochs_smooth, train_smooth, 'b-', label='Smoothed Training', linewidth=2)
        axes[1].plot(epochs_smooth, val_smooth, 'r-', label='Smoothed Validation', linewidth=2)
        axes[1].set_title('Smoothed Training Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def create_comparison_grid(original, reconstructed, num_samples=8):
    """
    Create a side-by-side comparison grid of original and reconstructed images.
    
    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        num_samples: Number of samples to show
        
    Returns:
        Comparison grid tensor
    """
    # Take only the requested number of samples
    original = original[:num_samples]
    reconstructed = reconstructed[:num_samples]
    
    # Denormalize
    original_denorm = denormalize_tensor(original)
    reconstructed_denorm = denormalize_tensor(reconstructed)
    
    # Clamp to valid range
    original_denorm = torch.clamp(original_denorm, 0, 1)
    reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)
    
    # Interleave original and reconstructed images
    comparison = torch.zeros(num_samples * 2, *original.shape[1:])
    comparison[0::2] = original_denorm  # Even indices: original
    comparison[1::2] = reconstructed_denorm  # Odd indices: reconstructed
    
    # Create grid
    grid = make_grid(comparison, nrow=2, padding=2, normalize=False)
    
    return grid


def analyze_latent_dimensions(model, dataloader, device, num_samples=1000):
    """
    Analyze the variance and usage of different latent dimensions.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader
        device: Device to run inference on
        num_samples: Number of samples to analyze
        
    Returns:
        dict: Analysis results
    """
    model.eval()
    
    latent_vectors = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_data in dataloader:
            if sample_count >= num_samples:
                break
            
            # Extract images from the batch
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                batch = batch_data[0].to(device)  # Get images (first element)
            else:
                batch = batch_data.to(device)  # Fallback for old format
            
            # Get latent representation
            if hasattr(model, 'encode'):
                if isinstance(model.encode(batch), tuple):  # VAE case
                    latent, _ = model.encode(batch)
                else:
                    latent = model.encode(batch)
            else:
                latent = model.encoder(batch)
            
            latent_vectors.append(latent.cpu().numpy())
            sample_count += batch.size(0)
    
    # Concatenate and analyze
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    
    # Compute statistics
    means = np.mean(latent_vectors, axis=0)
    stds = np.std(latent_vectors, axis=0)
    variances = np.var(latent_vectors, axis=0)
    
    # Find most and least active dimensions
    most_active = np.argsort(variances)[-10:][::-1]  # Top 10
    least_active = np.argsort(variances)[:10]  # Bottom 10
    
    results = {
        'means': means,
        'stds': stds,
        'variances': variances,
        'most_active_dims': most_active,
        'least_active_dims': least_active,
        'total_variance': np.sum(variances),
        'effective_dims': np.sum(variances > 0.01)  # Dimensions with variance > 0.01
    }
    
    return results 