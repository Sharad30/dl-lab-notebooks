import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicAutoencoder(nn.Module):
    """Basic fully connected autoencoder."""
    
    def __init__(self, input_dim=3*64*64, hidden_dims=[1024, 512, 256], latent_dim=128):
        """
        Args:
            input_dim (int): Input dimension (flattened image)
            hidden_dims (list): List of hidden layer dimensions
            latent_dim (int): Latent space dimension
        """
        super(BasicAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(in_dim, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space."""
        x = x.view(x.size(0), -1)  # Flatten
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to output."""
        x = self.decoder(z)
        return x.view(x.size(0), 3, 64, 64)  # Reshape to image
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        return self.decode(z)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for better image processing."""
    
    def __init__(self, input_channels=3, latent_dim=512):
        """
        Args:
            input_channels (int): Number of input channels
            latent_dim (int): Latent space dimension
        """
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            # Flatten and project to latent space
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Project from latent space and reshape
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),  # 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 3 x 64 x 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to output."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        return self.decode(z)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE) with convolutional layers."""
    
    def __init__(self, input_channels=3, latent_dim=256):
        """
        Args:
            input_channels (int): Number of input channels
            latent_dim (int): Latent space dimension
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Flatten()
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),  # 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 3 x 64 x 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space parameters."""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space to output."""
        h = self.decoder_fc(z)
        h = F.relu(h)
        return self.decoder_conv(h)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ResidualBlock(nn.Module):
    """Residual block for deeper autoencoder architectures."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResNetAutoencoder(nn.Module):
    """ResNet-style autoencoder with residual connections."""
    
    def __init__(self, input_channels=3, latent_dim=512, num_residual_blocks=2):
        """
        Args:
            input_channels (int): Number of input channels
            latent_dim (int): Latent space dimension
            num_residual_blocks (int): Number of residual blocks in each stage
        """
        super(ResNetAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Encoder residual blocks
        self.encoder_blocks = nn.ModuleList()
        channels = [64, 128, 256, 512]
        current_channels = 64
        
        for i, out_channels in enumerate(channels[1:]):
            # Downsample
            downsample = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.encoder_blocks.append(downsample)
            
            # Residual blocks
            for _ in range(num_residual_blocks):
                self.encoder_blocks.append(ResidualBlock(out_channels))
            
            current_channels = out_channels
        
        # Latent space projection
        self.encoder_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_initial = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4))
        )
        
        # Decoder residual blocks
        self.decoder_blocks = nn.ModuleList()
        channels = [512, 256, 128, 64]
        current_channels = 512
        
        for i, out_channels in enumerate(channels[1:]):
            # Residual blocks
            for _ in range(num_residual_blocks):
                self.decoder_blocks.append(ResidualBlock(current_channels))
            
            # Upsample
            upsample = nn.Sequential(
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.decoder_blocks.append(upsample)
            
            current_channels = out_channels
        
        # Final output layer
        self.decoder_final = nn.Sequential(
            nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        x = self.encoder_initial(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_final(x)
    
    def decode(self, z):
        """Decode from latent space to output."""
        x = self.decoder_initial(z)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_final(x)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        return self.decode(z)


def get_model(model_type='conv', **kwargs):
    """
    Factory function to create autoencoder models.
    
    Args:
        model_type (str): Type of autoencoder ('basic', 'conv', 'vae', 'resnet')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        torch.nn.Module: Autoencoder model
    """
    if model_type == 'basic':
        return BasicAutoencoder(**kwargs)
    elif model_type == 'conv':
        return ConvAutoencoder(**kwargs)
    elif model_type == 'vae':
        return VariationalAutoencoder(**kwargs)
    elif model_type == 'resnet':
        return ResNetAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 