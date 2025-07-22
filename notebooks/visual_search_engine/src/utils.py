import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List, Tuple, Optional, Dict
import pickle
from sklearn.metrics import average_precision_score
import requests
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


def get_transforms(phase: str = 'train'):
    """Get image transforms for TinyImageNet."""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def download_tiny_imagenet(data_dir: str):
    """Download and extract TinyImageNet dataset."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")
    
    if os.path.exists(extract_path):
        print("TinyImageNet already exists!")
        return extract_path
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Download
    print("Downloading TinyImageNet...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.remove(zip_path)
    return extract_path


class ResNetEmbedder(nn.Module):
    """ResNet101 feature extractor for generating 2048-dim embeddings."""
    
    def __init__(self, pretrained: bool = True):
        super(ResNetEmbedder, self).__init__()
        self.resnet = models.resnet101(pretrained=pretrained)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        """Extract features from input images."""
        with torch.no_grad():
            features = self.features(x)
            # Flatten to 2048-dim vector
            features = features.view(features.size(0), -1)
        return features


def extract_embeddings(model: ResNetEmbedder, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, List[int], List[str]]:
    """Extract embeddings for all images in the dataset."""
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"Extracted {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
    
    return embeddings_array, all_labels, all_paths


class FAISSIndex:
    """FAISS index for efficient similarity search."""
    
    def __init__(self, dimension: int = 2048):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.labels = None
        self.paths = None
        
    def build_index(self, embeddings: np.ndarray, labels: List[int], paths: List[str]):
        """Build FAISS index from embeddings."""
        print("Building FAISS index...")
        
        # Normalize embeddings for better similarity search
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings_normalized.astype('float32'))
        self.labels = np.array(labels)
        self.paths = paths
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k most similar images."""
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        distances, indices = self.index.search(query_normalized.astype('float32'), k)
        return distances[0], indices[0]
    
    def save(self, filepath: str):
        """Save index and metadata."""
        faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump({'labels': self.labels, 'paths': self.paths}, f)
    
    def load(self, filepath: str):
        """Load index and metadata."""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.meta", 'rb') as f:
            meta = pickle.load(f)
            self.labels = meta['labels']
            self.paths = meta['paths']


def calculate_precision_at_k(query_labels: List[int], retrieved_labels: List[List[int]], k: int) -> float:
    """Calculate Precision@k."""
    precisions = []
    for i, query_label in enumerate(query_labels):
        retrieved_k = retrieved_labels[i][:k]
        relevant_count = sum(1 for label in retrieved_k if label == query_label)
        precision = relevant_count / k
        precisions.append(precision)
    
    return np.mean(precisions)


def calculate_recall_at_k(query_labels: List[int], retrieved_labels: List[List[int]], 
                         all_labels: np.ndarray, k: int) -> float:
    """Calculate Recall@k."""
    recalls = []
    for i, query_label in enumerate(query_labels):
        retrieved_k = retrieved_labels[i][:k]
        relevant_count = sum(1 for label in retrieved_k if label == query_label)
        
        # Total relevant items in the dataset for this class
        total_relevant = np.sum(all_labels == query_label)
        
        recall = relevant_count / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)
    
    return np.mean(recalls)


def calculate_map(query_labels: List[int], retrieved_labels: List[List[int]], k: int = None) -> float:
    """Calculate Mean Average Precision (mAP)."""
    aps = []
    
    for i, query_label in enumerate(query_labels):
        retrieved = retrieved_labels[i] if k is None else retrieved_labels[i][:k]
        
        # Binary relevance scores
        relevance = [1 if label == query_label else 0 for label in retrieved]
        
        if sum(relevance) == 0:
            aps.append(0.0)
            continue
        
        # Calculate AP
        precisions = []
        relevant_count = 0
        for j, rel in enumerate(relevance):
            if rel == 1:
                relevant_count += 1
                precision = relevant_count / (j + 1)
                precisions.append(precision)
        
        ap = np.mean(precisions) if precisions else 0.0
        aps.append(ap)
    
    return np.mean(aps)


def visualize_search_results(query_path: str, retrieved_paths: List[str], 
                           distances: np.ndarray, k: int = 5, figsize: Tuple[int, int] = (15, 3)):
    """Visualize query image and top-k retrieved results."""
    fig, axes = plt.subplots(1, k + 1, figsize=figsize)
    
    # Query image
    query_img = Image.open(query_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Retrieved images
    for i in range(k):
        if i < len(retrieved_paths):
            img = Image.open(retrieved_paths[i]).convert('RGB')
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f'#{i+1}\nDist: {distances[i]:.3f}', fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_search_engine(model: ResNetEmbedder, index: FAISSIndex, 
                          val_dataset: TinyImageNetDataset, device: str, 
                          num_queries: int = 100, k_values: List[int] = [1, 5, 10]) -> Dict:
    """Comprehensive evaluation of the search engine."""
    model.eval()
    model.to(device)
    
    # Sample random queries from validation set
    query_indices = np.random.choice(len(val_dataset), num_queries, replace=False)
    
    results = {}
    all_retrieved_labels = []
    query_labels = []
    
    print(f"Evaluating with {num_queries} queries...")
    
    for idx in tqdm(query_indices):
        image, label, path = val_dataset[idx]
        query_labels.append(label)
        
        # Extract query embedding
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            query_embedding = model(image_batch).cpu().numpy()
        
        # Search
        distances, indices = index.search(query_embedding, k=max(k_values))
        retrieved_labels = [index.labels[i] for i in indices]
        all_retrieved_labels.append(retrieved_labels)
    
    # Calculate metrics
    for k in k_values:
        precision_k = calculate_precision_at_k(query_labels, all_retrieved_labels, k)
        recall_k = calculate_recall_at_k(query_labels, all_retrieved_labels, index.labels, k)
        map_k = calculate_map(query_labels, all_retrieved_labels, k)
        
        results[f'precision@{k}'] = precision_k
        results[f'recall@{k}'] = recall_k
        results[f'map@{k}'] = map_k
        
        print(f"Precision@{k}: {precision_k:.4f}")
        print(f"Recall@{k}: {recall_k:.4f}")
        print(f"mAP@{k}: {map_k:.4f}")
        print("-" * 30)
    
    return results


def search_and_visualize(model: ResNetEmbedder, index: FAISSIndex, 
                        query_path: str, device: str, k: int = 5):
    """Search for similar images and visualize results."""
    transform = get_transforms('val')
    
    # Load and preprocess query image
    query_image = Image.open(query_path).convert('RGB')
    query_tensor = transform(query_image).unsqueeze(0).to(device)
    
    # Extract embedding
    model.eval()
    with torch.no_grad():
        query_embedding = model(query_tensor).cpu().numpy()
    
    # Search
    distances, indices = index.search(query_embedding, k)
    retrieved_paths = [index.paths[i] for i in indices]
    
    # Visualize
    visualize_search_results(query_path, retrieved_paths, distances, k)
    
    return retrieved_paths, distances, indices 