import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from typing import Dict, Tuple, Optional, Any
import torch.nn.functional as F
from torchvision import models, transforms
from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity

# Try to import optional dependencies for advanced metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics")


class AutoencoderTrainer:
    """Comprehensive trainer for autoencoder models."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_type='standard'):
        """
        Args:
            model: Autoencoder model
            device: Device to train on
            model_type: Type of model ('standard', 'vae')
        """
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = None
        self.log_dir = None
        
        # Initialize evaluation metrics
        self._init_evaluation_metrics()
        
    def _init_evaluation_metrics(self):
        """Initialize evaluation metrics if available."""
        self.lpips_metric = None
        self.ssim_metric = None
        self.inception_model = None
        
        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            try:
                self.lpips_metric = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_metric.eval()
            except Exception as e:
                print(f"Warning: Could not initialize LPIPS: {e}")
        
        # Initialize SSIM
        if SSIM_AVAILABLE:
            try:
                self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            except Exception as e:
                print(f"Warning: Could not initialize SSIM: {e}")
        
        # Initialize Inception model for FID
        try:
            self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()
            # Remove the final classification layer for feature extraction
            self.inception_model.fc = nn.Identity()  # type: ignore
        except Exception as e:
            print(f"Warning: Could not initialize Inception model for FID: {e}")
            self.inception_model = None
        
    def setup_logging(self, log_dir=None):
        """Setup tensorboard logging."""
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/autoencoder_{timestamp}"
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def compute_loss(self, x, output, mu=None, logvar=None):
        """
        Compute loss based on model type.
        
        Args:
            x: Input images
            output: Reconstructed images
            mu: Mean for VAE (optional)
            logvar: Log variance for VAE (optional)
            
        Returns:
            Total loss and loss components
        """
        # Reconstruction loss
        recon_loss = nn.MSELoss()(output, x)
        
        if self.model_type == 'vae' and mu is not None and logvar is not None:
            # KL divergence loss for VAE
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            total_loss = recon_loss + 0.001 * kl_loss  # Beta = 0.001
            return total_loss, {'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}
        else:
            return recon_loss, {'recon_loss': recon_loss.item()}
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch_data in enumerate(pbar):
            # Extract images from batch using helper method
            try:
                data = self._extract_images_from_batch(batch_data)
            except Exception as e:
                print(f"Warning: Could not extract images from batch: {e}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'vae':
                try:
                    # Try standard VAE interface
                    output, mu, logvar = self.model(data)
                    loss, components = self.compute_loss(data, output, mu, logvar)
                except ValueError:
                    # Fallback for different VAE interfaces
                    model_output = self.model(data)
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        output = model_output[0]
                        mu = model_output[1]
                        loss, components = self.compute_loss(data, output)
                    else:
                        output = model_output
                        print("Warning: Could not extract latent variables from VAE output")
            else:
                output = self.model(data)
                # For standard autoencoder, try to get latent representation
                try:
                    if hasattr(self.model, 'encoder'):
                        latent = self.model.encoder(data)
                        loss, components = self.compute_loss(data, output)
                    elif hasattr(self.model, 'encode'):
                        latent = self.model.encode(data)
                        loss, components = self.compute_loss(data, output)
                except Exception as e:
                    print(f"Warning: Could not extract latent representation: {e}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if self.writer and batch_idx % 100 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Extract images from batch using helper method
                try:
                    data = self._extract_images_from_batch(batch_data)
                except Exception as e:
                    print(f"Warning: Could not extract images from batch: {e}")
                    continue
                
                # Forward pass
                if self.model_type == 'vae':
                    output, mu, logvar = self.model(data)
                    loss, components = self.compute_loss(data, output, mu, logvar)
                else:
                    output = self.model(data)
                    loss, components = self.compute_loss(data, output)
                
                total_loss += loss.item()
                
                # Accumulate loss components
                for key, value in components.items():
                    if key not in loss_components:
                        loss_components[key] = 0
                    loss_components[key] += value
        
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, 
              optimizer_type='adam', scheduler_type=None, 
              save_dir='checkpoints', save_every=10):
        """
        Train the autoencoder.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            scheduler_type: Type of learning rate scheduler ('step', 'cosine', None)
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=1e-5)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup scheduler
        scheduler = None
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Setup logging
        if self.writer is None:
            self.setup_logging()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training autoencoder for {epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader, optimizer, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_components = self.validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                
                # Log loss components
                for key, value in train_components.items():
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
                for key, value in val_components.items():
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'), 
                                   optimizer, epoch, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                                   optimizer, epoch)
        
        # Save final model
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pth'), 
                           optimizer, epochs-1, is_final=True)
        
        if self.writer:
            self.writer.close()
        
        print("Training completed!")
        
    def save_checkpoint(self, filepath, optimizer, epoch, is_best=False, is_final=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_type': self.model_type
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Best model saved at epoch {epoch+1} with validation loss: {self.best_val_loss:.4f}")
        elif is_final:
            print(f"Final model saved after {epoch+1} epochs")
    
    def load_checkpoint(self, filepath, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from epoch {self.current_epoch+1}")
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _extract_images_from_batch(self, batch_data) -> torch.Tensor:
        """Extract images from batch data, handling different formats."""
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            return batch_data[0].to(self.device)  # Get images (first element)
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.to(self.device)  # Direct tensor
        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")

    def compute_ssim(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Structural Similarity Index."""
        if self.ssim_metric is None:
            # Fallback manual SSIM implementation
            return self._manual_ssim(original, reconstructed)
        
        try:
            # Ensure values are in [0, 1] range
            original = torch.clamp(original, 0, 1)
            reconstructed = torch.clamp(reconstructed, 0, 1)
            return self.ssim_metric(reconstructed, original).item()
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            return 0.0
    
    def _manual_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Manual SSIM implementation as fallback."""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean().item()

    def compute_lpips(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Learned Perceptual Image Patch Similarity."""
        if self.lpips_metric is None:
            return 0.0
        
        try:
            # LPIPS expects values in [-1, 1] range
            original_norm = original * 2.0 - 1.0
            reconstructed_norm = reconstructed * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_score = self.lpips_metric(original_norm, reconstructed_norm)
            return lpips_score.mean().item()
        except Exception as e:
            print(f"Warning: LPIPS computation failed: {e}")
            return 0.0

    def compute_fid_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for FID computation."""
        if self.inception_model is None:
            return torch.zeros(images.size(0), 2048)
        
        try:
            # Resize images to 299x299 for Inception
            if images.size(-1) != 299:
                images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Ensure 3 channels
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            with torch.no_grad():
                features = self.inception_model(images)
            return features
        except Exception as e:
            print(f"Warning: FID feature extraction failed: {e}")
            return torch.zeros(images.size(0), 2048)

    def compute_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance."""
        try:
            # Convert to numpy
            real_features = real_features.cpu().numpy()
            fake_features = fake_features.cpu().numpy()
            
            # Compute statistics
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
            
            # Compute FID
            diff = mu1 - mu2
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid)
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            return float('inf')

    def compute_comprehensive_metrics(self, data_loader, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            data_loader: Data loader for evaluation
            num_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary containing all computed metrics
        """
        self.model.eval()
        
        # Initialize metric accumulators
        mse_scores = []
        ssim_scores = []
        lpips_scores = []
        psnr_scores = []
        
        # For FID computation
        real_features = []
        fake_features = []
        
        # For latent analysis
        latent_vectors = []
        
        sample_count = 0
        
        print(f"Computing comprehensive metrics on {min(num_samples, len(data_loader.dataset))} samples...")
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Evaluating"):
                if sample_count >= num_samples:
                    break
                
                # Extract images from batch
                try:
                    data = self._extract_images_from_batch(batch_data)
                except Exception as e:
                    print(f"Warning: Could not extract images from batch: {e}")
                    continue
                
                batch_size = data.size(0)
                if sample_count + batch_size > num_samples:
                    data = data[:num_samples - sample_count]
                    batch_size = data.size(0)
                
                # Forward pass
                if self.model_type == 'vae':
                    try:
                        # Try standard VAE interface
                        output, mu, logvar = self.model(data)
                        latent_vectors.append(mu.cpu().numpy())
                    except ValueError:
                        # Fallback for different VAE interfaces
                        model_output = self.model(data)
                        if isinstance(model_output, tuple) and len(model_output) >= 2:
                            output = model_output[0]
                            mu = model_output[1]
                            latent_vectors.append(mu.cpu().numpy())
                        else:
                            output = model_output
                            print("Warning: Could not extract latent variables from VAE output")
                else:
                    output = self.model(data)
                    # For standard autoencoder, try to get latent representation
                    try:
                        if hasattr(self.model, 'encoder'):
                            latent = self.model.encoder(data)
                            latent_vectors.append(latent.cpu().numpy())
                        elif hasattr(self.model, 'encode'):
                            latent = self.model.encode(data)
                            latent_vectors.append(latent.cpu().numpy())
                    except Exception as e:
                        print(f"Warning: Could not extract latent representation: {e}")
                
                # Ensure values are in [0, 1] range
                data_clamped = torch.clamp(data, 0, 1)
                output_clamped = torch.clamp(output, 0, 1)
                
                # Compute MSE
                mse = F.mse_loss(output_clamped, data_clamped, reduction='none')
                mse = mse.view(batch_size, -1).mean(dim=1)
                mse_scores.extend(mse.cpu().numpy())
                
                # Compute PSNR
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                psnr_scores.extend(psnr.cpu().numpy())
                
                # Compute SSIM
                for i in range(batch_size):
                    ssim_score = self.compute_ssim(data_clamped[i:i+1], output_clamped[i:i+1])
                    ssim_scores.append(ssim_score)
                
                # Compute LPIPS
                lpips_score = self.compute_lpips(data_clamped, output_clamped)
                lpips_scores.extend([lpips_score] * batch_size)
                
                # Extract features for FID
                real_feat = self.compute_fid_features(data_clamped)
                fake_feat = self.compute_fid_features(output_clamped)
                real_features.append(real_feat)
                fake_features.append(fake_feat)
                
                sample_count += batch_size
        
        # Compute FID
        if real_features and fake_features:
            real_features_all = torch.cat(real_features, dim=0)
            fake_features_all = torch.cat(fake_features, dim=0)
            fid_score = self.compute_fid(real_features_all, fake_features_all)
        else:
            fid_score = float('inf')
        
        # Analyze latent space
        latent_analysis = {}
        if latent_vectors:
            latent_all = np.concatenate(latent_vectors, axis=0)
            latent_analysis = {
                'latent_dim': latent_all.shape[1],
                'mean_activation': np.mean(np.abs(latent_all)),
                'std_activation': np.std(latent_all),
                'active_dimensions': np.sum(np.var(latent_all, axis=0) > 0.01),
                'total_variance': np.sum(np.var(latent_all, axis=0))
            }
        
        # Compile results
        metrics = {
            'reconstruction_metrics': {
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'psnr_mean': np.mean(psnr_scores),
                'psnr_std': np.std(psnr_scores),
            },
            'perceptual_metrics': {
                'ssim_mean': np.mean(ssim_scores),
                'ssim_std': np.std(ssim_scores),
                'lpips_mean': np.mean(lpips_scores),
                'lpips_std': np.std(lpips_scores),
            },
            'distributional_metrics': {
                'fid_score': fid_score,
            },
            'latent_analysis': latent_analysis,
            'sample_count': sample_count
        }
        
        return metrics

    def post_training_evaluation(self, val_loader, test_loader=None, 
                               save_dir: str = 'evaluation_results',
                               num_samples: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive post-training evaluation on validation and test data.
        
        Args:
            val_loader: Validation data loader
            test_loader: Optional test data loader for unseen data evaluation
            save_dir: Directory to save evaluation results
            num_samples: Maximum number of samples to evaluate per dataset
            
        Returns:
            Dictionary containing all evaluation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting comprehensive post-training evaluation...")
        
        # Evaluate on validation data
        print("\n" + "="*50)
        print("VALIDATION SET EVALUATION")
        print("="*50)
        val_metrics = self.compute_comprehensive_metrics(val_loader, num_samples)
        
        # Evaluate on test data if provided
        test_metrics = None
        if test_loader is not None:
            print("\n" + "="*50)
            print("TEST SET EVALUATION (Unseen Data)")
            print("="*50)
            test_metrics = self.compute_comprehensive_metrics(test_loader, num_samples)
        
        # Print comprehensive results
        self._print_evaluation_results(val_metrics, test_metrics)
        
        # Save results to file
        results = {
            'model_type': self.model_type,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'evaluation_timestamp': datetime.now().isoformat(),
        }
        
        results_file = os.path.join(save_dir, 'comprehensive_evaluation.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEvaluation results saved to: {results_file}")
        
        # Create comparison plots
        self._create_evaluation_plots(val_metrics, test_metrics, save_dir)
        
        return results
    
    def _print_evaluation_results(self, val_metrics: Dict[str, Any], 
                                test_metrics: Optional[Dict[str, Any]] = None):
        """Print formatted evaluation results."""
        
        def print_metrics_section(title: str, metrics: Dict[str, Any]):
            print(f"\n{title}:")
            print("-" * len(title))
            
            # Reconstruction metrics
            recon = metrics['reconstruction_metrics']
            print(f"MSE:  {recon['mse_mean']:.6f} ± {recon['mse_std']:.6f}")
            print(f"PSNR: {recon['psnr_mean']:.2f} ± {recon['psnr_std']:.2f} dB")
            
            # Perceptual metrics
            percep = metrics['perceptual_metrics']
            print(f"SSIM: {percep['ssim_mean']:.4f} ± {percep['ssim_std']:.4f}")
            print(f"LPIPS: {percep['lpips_mean']:.4f} ± {percep['lpips_std']:.4f}")
            
            # Distributional metrics
            distrib = metrics['distributional_metrics']
            if distrib['fid_score'] != float('inf'):
                print(f"FID:  {distrib['fid_score']:.2f}")
            else:
                print("FID:  Not computed (missing dependencies)")
            
            # Latent analysis
            if metrics['latent_analysis']:
                latent = metrics['latent_analysis']
                print(f"\nLatent Space Analysis:")
                print(f"  Dimensions: {latent['latent_dim']}")
                print(f"  Active dims: {latent['active_dimensions']}")
                print(f"  Mean activation: {latent['mean_activation']:.4f}")
                print(f"  Total variance: {latent['total_variance']:.4f}")
            
            print(f"Samples evaluated: {metrics['sample_count']}")
        
        # Print validation results
        print_metrics_section("VALIDATION METRICS", val_metrics)
        
        # Print test results if available
        if test_metrics is not None:
            print_metrics_section("TEST METRICS (Unseen Data)", test_metrics)
            
            # Print comparison
            print(f"\nVALIDATION vs TEST COMPARISON:")
            print("-" * 30)
            val_mse = val_metrics['reconstruction_metrics']['mse_mean']
            test_mse = test_metrics['reconstruction_metrics']['mse_mean']
            print(f"MSE degradation: {((test_mse - val_mse) / val_mse * 100):.2f}%")
            
            val_ssim = val_metrics['perceptual_metrics']['ssim_mean']
            test_ssim = test_metrics['perceptual_metrics']['ssim_mean']
            print(f"SSIM degradation: {((val_ssim - test_ssim) / val_ssim * 100):.2f}%")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _create_evaluation_plots(self, val_metrics: Dict[str, Any], 
                               test_metrics: Optional[Dict[str, Any]], 
                               save_dir: str):
        """Create evaluation comparison plots."""
        
        # Create metrics comparison plot
        metrics_to_plot = ['mse_mean', 'psnr_mean', 'ssim_mean', 'lpips_mean']
        val_values = [
            val_metrics['reconstruction_metrics']['mse_mean'],
            val_metrics['reconstruction_metrics']['psnr_mean'],
            val_metrics['perceptual_metrics']['ssim_mean'],
            val_metrics['perceptual_metrics']['lpips_mean']
        ]
        
        if test_metrics is not None:
            test_values = [
                test_metrics['reconstruction_metrics']['mse_mean'],
                test_metrics['reconstruction_metrics']['psnr_mean'],
                test_metrics['perceptual_metrics']['ssim_mean'],
                test_metrics['perceptual_metrics']['lpips_mean']
            ]
            
            # Normalize values for comparison (since they have different scales)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            metric_names = ['MSE', 'PSNR (dB)', 'SSIM', 'LPIPS']
            
            for i, (name, val_val, test_val) in enumerate(zip(metric_names, val_values, test_values)):
                axes[i].bar(['Validation', 'Test'], [val_val, test_val], alpha=0.7)
                axes[i].set_title(f'{name}')
                axes[i].set_ylabel(name)
                
                # Add value labels on bars
                axes[i].text(0, val_val, f'{val_val:.4f}', ha='center', va='bottom')
                axes[i].text(1, test_val, f'{test_val:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Evaluation plots saved to: {save_dir}")


def create_trainer(model, model_type='standard', **kwargs):
    """
    Factory function to create a trainer.
    
    Args:
        model: Autoencoder model
        model_type: Type of model ('standard', 'vae')
        **kwargs: Additional arguments for trainer
        
    Returns:
        AutoencoderTrainer: Configured trainer
    """
    return AutoencoderTrainer(model, model_type=model_type, **kwargs)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False 