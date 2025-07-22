import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
from tqdm import tqdm


class TinyImageNetDataset(Dataset):
    """Dataset class for TinyImageNet with proper normalization."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Path to TinyImageNet root directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset paths and labels."""
        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls in classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                if os.path.exists(cls_dir):
                    for img_name in os.listdir(cls_dir):
                        if img_name.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(cls_dir, img_name))
                            self.labels.append(self.class_to_idx[cls])
        
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val')
            images_dir = os.path.join(val_dir, 'images')
            annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            
            # Load class names from train directory
            train_dir = os.path.join(self.root_dir, 'train')
            classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Parse annotations
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    self.image_paths.append(os.path.join(images_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_transforms(phase: str = 'train', image_size: int = 64):
    """Get image transforms for TinyImageNet."""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        
def get_tinyimagenet_dataloaders(root_dir='./data', batch_size=64, num_workers=4, 
                                download=True, image_size=64):
    """
    Create DataLoaders for TinyImageNet dataset.
    
    Args:
        root_dir (str): Root directory to store/load the dataset
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        download (bool): Whether to download the dataset if not present
        image_size (int): Size to resize images to
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Use the consistent transforms with the specified image_size
    train_dataset = TinyImageNetDataset(
        "../../data/tiny-imagenet-200", 
        'train', 
        get_transforms('train', image_size)
    )
    val_dataset = TinyImageNetDataset(
        "../../data/tiny-imagenet-200", 
        'val', 
        get_transforms('val', image_size)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean 