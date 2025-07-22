"""
Face Search Extension for Visual Search Engine

This module extends the visual search engine to work with face datasets
like CelebA and VGGFace2, including face alignment and identity-based evaluation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
# import face_recognition  # Optional: pip install face_recognition
from tqdm import tqdm
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from utils import FAISSIndex


class FaceAlignmentTransform:
    """Face alignment preprocessing for better face recognition."""
    
    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        self.output_size = output_size
    
    def __call__(self, image):
        """Align face and resize to output_size."""
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        try:
            import face_recognition
            # Find face landmarks
            face_landmarks_list = face_recognition.face_landmarks(image_np)
            
            if not face_landmarks_list:
                # No face found, return center crop
                image_pil = Image.fromarray(image_np) if not isinstance(image, Image.Image) else image
                return transforms.Resize(self.output_size)(image_pil)
            
            # Use first detected face
            face_landmarks = face_landmarks_list[0]
            
            # Get eye positions for alignment
            left_eye = np.mean(face_landmarks['left_eye'], axis=0)
            right_eye = np.mean(face_landmarks['right_eye'], axis=0)
            
            # Calculate angle and center
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate image
            h, w = image_np.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            aligned = cv2.warpAffine(image_np, rotation_matrix, (w, h))
            
            # Convert back to PIL and resize
            aligned_pil = Image.fromarray(aligned)
            return transforms.Resize(self.output_size)(aligned_pil)
        except ImportError:
            # face_recognition not available, fall back to simple resize
            image_pil = Image.fromarray(image_np) if not isinstance(image, Image.Image) else image
            return transforms.Resize(self.output_size)(image_pil)


class CelebADataset(Dataset):
    """Dataset class for CelebA with attribute labels and face alignment."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, 
                 align_faces: bool = False, max_samples: Optional[int] = None, 
                 target_attribute: str = 'Male'):
        """
        Args:
            root_dir: Path to CelebA root directory (should contain img_align_celeba folder)
            split: 'train', 'val', or 'test'
            transform: Image transformations
            align_faces: Whether to apply face alignment (requires face_recognition package)
            max_samples: Limit number of samples (for prototyping)
            target_attribute: Which attribute to use as label (default: 'Male' for gender classification)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.align_faces = align_faces
        self.target_attribute = target_attribute
        
        self.image_paths = []
        self.labels = []
        self.all_attributes = []
        
        if align_faces:
            self.face_align = FaceAlignmentTransform()
        
        self._load_dataset(max_samples)
    
    def _load_dataset(self, max_samples: Optional[int] = None):
        """Load CelebA dataset with attribute information."""
        # File paths
        images_dir = os.path.join(self.root_dir, 'img_align_celeba')
        split_file = os.path.join(self.root_dir, 'list_eval_partition.csv')
        attr_file = os.path.join(self.root_dir, 'list_attr_celeba.csv')
        
        # Check if files exist
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        if not os.path.exists(attr_file):
            raise FileNotFoundError(f"Attributes file not found: {attr_file}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Map split names to numbers
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map[self.split]
        
        # Load split information
        print("Loading split information...")
        split_df = pd.read_csv(split_file)
        split_info = dict(zip(split_df['image_id'], split_df['partition']))
        
        # Load attributes information
        print("Loading attributes information...")
        attr_df = pd.read_csv(attr_file)
        
        # Check if target attribute exists
        if self.target_attribute not in attr_df.columns:
            available_attrs = [col for col in attr_df.columns if col != 'image_id']
            raise ValueError(f"Target attribute '{self.target_attribute}' not found. "
                           f"Available attributes: {available_attrs[:10]}...")
        
        print(f"Using '{self.target_attribute}' as target attribute")
        
        # Filter by split and load data
        count = 0
        for _, row in attr_df.iterrows():
            img_name = str(row['image_id'])  # Convert to string explicitly
            
            # Check if this image belongs to our split
            if split_info.get(img_name) == target_split:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    
                    # Convert attribute from {-1, 1} to {0, 1}
                    attr_value = row[self.target_attribute]
                    label = 1 if attr_value == 1 else 0
                    self.labels.append(label)
                    
                    # Store all attributes for potential future use
                    attrs = row.drop('image_id').values
                    self.all_attributes.append(attrs)
                    
                    count += 1
                    
                    # Limit samples for prototyping
                    if max_samples and count >= max_samples:
                        break
        
        print(f"Loaded {len(self.image_paths)} images from {self.split} split")
        print(f"Target attribute '{self.target_attribute}' distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count} images ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply face alignment if requested
        if self.align_faces:
            try:
                image = self.face_align(image)
            except:
                # Fall back to center crop if alignment fails
                image = transforms.Resize((224, 224))(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_face_transforms(phase: str = 'train'):
    """Get image transforms optimized for face recognition."""
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    if phase == 'train':
        # Add augmentations for training
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
        ] + base_transforms)
    else:
        return transforms.Compose(base_transforms)


def evaluate_face_search(model, index, val_dataset, device: str, 
                        num_queries: int = 100, k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate face search with attribute matching.
    
    For CelebA, we evaluate based on the target attribute (e.g., gender, age, etc.)
    """
    model.eval()
    model.to(device)
    
    # Sample random queries
    query_indices = np.random.choice(len(val_dataset), num_queries, replace=False)
    
    results = {}
    all_retrieved_labels = []
    query_labels = []
    
    print(f"Evaluating face search with {num_queries} queries...")
    
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
    
    # Calculate attribute-based metrics
    for k in k_values:
        # Attribute Precision@k: fraction of retrieved faces with same attribute
        attr_precisions = []
        for i, query_label in enumerate(query_labels):
            retrieved_k = all_retrieved_labels[i][:k]
            same_attr_count = sum(1 for label in retrieved_k if label == query_label)
            precision = same_attr_count / k
            attr_precisions.append(precision)
        
        avg_attr_precision = np.mean(attr_precisions)
        results[f'attribute_precision@{k}'] = avg_attr_precision
        
        # Attribute Recall@k: fraction of same-attribute faces found
        attr_recalls = []
        for i, query_label in enumerate(query_labels):
            retrieved_k = all_retrieved_labels[i][:k]
            same_attr_count = sum(1 for label in retrieved_k if label == query_label)
            
            # Total same-attribute faces in database
            total_same_attr = np.sum(np.array(index.labels) == query_label)
            recall = same_attr_count / total_same_attr if total_same_attr > 0 else 0
            attr_recalls.append(recall)
        
        avg_attr_recall = np.mean(attr_recalls)
        results[f'attribute_recall@{k}'] = avg_attr_recall
        
        # Attribute mAP@k: Mean Average Precision for attribute matching
        aps = []
        for i, query_label in enumerate(query_labels):
            retrieved_k = all_retrieved_labels[i][:k]
            
            # Binary relevance scores for attribute matching
            relevance = [1 if label == query_label else 0 for label in retrieved_k]
            
            if sum(relevance) == 0:
                aps.append(0.0)
                continue
            
            # Calculate AP for this query
            precisions = []
            relevant_count = 0
            for j, rel in enumerate(relevance):
                if rel == 1:
                    relevant_count += 1
                    precision = relevant_count / (j + 1)
                    precisions.append(precision)
            
            ap = np.mean(precisions) if precisions else 0.0
            aps.append(ap)
        
        avg_map = np.mean(aps)
        results[f'attribute_map@{k}'] = avg_map
        
        print(f"Attribute Precision@{k}: {avg_attr_precision:.4f}")
        print(f"Attribute Recall@{k}: {avg_attr_recall:.4f}")
        print(f"Attribute mAP@{k}: {avg_map:.4f}")
        print("-" * 30)
    
    return results


def calculate_face_verification_metrics(model, index, val_dataset, device: str, 
                                       num_pairs: int = 1000) -> Dict:
    """
    Calculate face verification metrics (same/different attribute classification).
    """
    model.eval()
    model.to(device)
    
    # Generate positive and negative pairs
    positive_pairs = []  # Same attribute
    negative_pairs = []  # Different attribute
    
    # Group images by attribute
    attr_groups = {0: [], 1: []}
    for i, (_, attr, _) in enumerate(val_dataset):
        attr_groups[attr].append(i)
    
    # Generate positive pairs (same attribute)
    for attr_val, indices in attr_groups.items():
        if len(indices) >= 2:
            num_pairs_per_attr = min(num_pairs // 4, len(indices) // 2)
            for i in range(num_pairs_per_attr):
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                positive_pairs.append((idx1, idx2))
    
    # Generate negative pairs (different attribute)
    attr_0_indices = attr_groups[0]
    attr_1_indices = attr_groups[1]
    
    for _ in range(len(positive_pairs)):
        if len(attr_0_indices) > 0 and len(attr_1_indices) > 0:
            idx1 = np.random.choice(attr_0_indices)
            idx2 = np.random.choice(attr_1_indices)
            negative_pairs.append((idx1, idx2))
    
    # Combine pairs
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    similarities = []
    
    print(f"Computing similarities for {len(all_pairs)} pairs...")
    for idx1, idx2 in tqdm(all_pairs):
        # Get embeddings
        image1, _, _ = val_dataset[idx1]
        image2, _, _ = val_dataset[idx2]
        
        with torch.no_grad():
            emb1 = model(image1.unsqueeze(0).to(device)).cpu().numpy()
            emb2 = model(image2.unsqueeze(0).to(device)).cpu().numpy()
        
        # Calculate cosine similarity
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(emb1_norm, emb2_norm.T)[0, 0]
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Find optimal threshold
    thresholds = np.linspace(similarities.min(), similarities.max(), 100)
    best_acc = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    
    # Calculate final metrics
    predictions = (similarities > best_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    
    # True/False Positive/Negative rates
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    results = {
        'verification_accuracy': accuracy,
        'true_positive_rate': tpr,
        'false_positive_rate': fpr,
        'optimal_threshold': best_threshold,
        'num_pairs': len(all_pairs)
    }
    
    print(f"Face Verification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"True Positive Rate: {tpr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    
    return results


def visualize_face_search_results(query_path: str, retrieved_paths: List[str], 
                                  distances: np.ndarray, query_attrs: Optional[Dict] = None,
                                  retrieved_attrs: Optional[List[Dict]] = None, k: int = 5, 
                                  figsize: Tuple[int, int] = (15, 4)):
    """
    Visualize face search results with attribute information.
    
    Args:
        query_path: Path to query face image
        retrieved_paths: List of paths to retrieved face images
        distances: Array of distances from query
        query_attrs: Dictionary of query face attributes
        retrieved_attrs: List of dictionaries with retrieved faces' attributes
        k: Number of results to show
        figsize: Figure size for matplotlib
    """
    fig, axes = plt.subplots(1, k + 1, figsize=figsize)
    
    # Query image
    query_img = Image.open(query_path).convert('RGB')
    axes[0].imshow(query_img)
    
    # Query title with attributes
    query_title = 'QUERY'
    if query_attrs:
        # Show key attributes
        key_attrs = []
        for attr, value in query_attrs.items():
            if attr in ['Male', 'Young', 'Smiling', 'Attractive', 'Eyeglasses']:
                key_attrs.append(f"{attr}: {'Yes' if value == 1 else 'No'}")
        if key_attrs:
            query_title += '\n' + '\n'.join(key_attrs[:3])  # Show top 3 attributes
    
    axes[0].set_title(query_title, fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Retrieved images
    for i in range(k):
        if i < len(retrieved_paths):
            img = Image.open(retrieved_paths[i]).convert('RGB')
            axes[i + 1].imshow(img)
            
            # Title with distance and attributes
            title = f'#{i+1}\nDist: {distances[i]:.3f}'
            
            if retrieved_attrs and i < len(retrieved_attrs):
                attrs = retrieved_attrs[i]
                # Show key attributes
                key_attrs = []
                for attr, value in attrs.items():
                    if attr in ['Male', 'Young', 'Smiling', 'Attractive', 'Eyeglasses']:
                        key_attrs.append(f"{attr}: {'Yes' if value == 1 else 'No'}")
                if key_attrs:
                    title += '\n' + '\n'.join(key_attrs[:2])  # Show top 2 attributes
            
            axes[i + 1].set_title(title, fontsize=8)
        else:
            axes[i + 1].set_title(f'#{i+1}\nNo result', fontsize=8)
        
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def face_search_and_visualize(model, index, query_path: str, 
                             face_dataset: CelebADataset, device: str, k: int = 5):
    """
    Search for similar faces and visualize results with attribute information.
    
    Args:
        model: ResNet embedder model
        index: FAISS search index
        query_path: Path to query face image
        face_dataset: CelebADataset instance for attribute lookup
        device: Device for model inference
        k: Number of similar faces to retrieve
    
    Returns:
        retrieved_paths: List of retrieved image paths
        distances: Array of distances
        indices: Array of retrieved indices
        query_attrs: Query face attributes
        retrieved_attrs: List of retrieved faces' attributes
    """
    from utils import get_transforms  # Import here to avoid circular import
    
    transform = get_transforms('val')
    
    # Load and preprocess query image
    query_image = Image.open(query_path).convert('RGB')
    query_tensor = transform(query_image)  # This returns a tensor
    query_batch = query_tensor.unsqueeze(0).to(device)  # Now we can unsqueeze
    
    # Extract embedding
    model.eval()
    with torch.no_grad():
        query_embedding = model(query_batch).cpu().numpy()
    
    # Search
    distances, indices = index.search(query_embedding, k)
    retrieved_paths = [index.paths[i] for i in indices]
    
    # Get attribute information
    query_attrs = None
    retrieved_attrs = []
    
    # Find query in dataset to get attributes
    query_filename = os.path.basename(query_path)
    for i in range(len(face_dataset)):
        _, _, path = face_dataset[i]
        if os.path.basename(path) == query_filename:
            # Get all attributes for this image
            if i < len(face_dataset.all_attributes):
                attr_values = face_dataset.all_attributes[i]
                # Create attribute dictionary (we need column names)
                # This is a simplified version - in practice you'd store column names
                query_attrs = {
                    'Male': face_dataset.labels[i] if face_dataset.target_attribute == 'Male' else None,
                    'Target': face_dataset.labels[i]
                }
            break
    
    # Get attributes for retrieved faces
    for idx in indices:
        if hasattr(index, 'dataset_indices'):
            # If we stored original dataset indices
            dataset_idx = index.dataset_indices[idx]
            if dataset_idx < len(face_dataset.all_attributes):
                attr_values = face_dataset.all_attributes[dataset_idx]
                retrieved_attrs.append({
                    'Target': index.labels[idx]
                })
        else:
            # Fallback: just use the target attribute
            retrieved_attrs.append({
                'Target': index.labels[idx]
            })
    
    # Visualize
    if query_attrs is not None:
        visualize_face_search_results(
            query_path, retrieved_paths, distances, 
            query_attrs, retrieved_attrs, k
        )
    else:
        visualize_face_search_results(
            query_path, retrieved_paths, distances, 
            None, retrieved_attrs, k
        )
    
    return retrieved_paths, distances, indices, query_attrs, retrieved_attrs


def demo_face_search(model, index, face_dataset: CelebADataset, 
                    device: str, num_demos: int = 3, k: int = 5):
    """
    Demo face search with random queries from the dataset.
    
    Args:
        model: ResNet embedder model
        index: FAISS search index  
        face_dataset: CelebADataset instance
        device: Device for model inference
        num_demos: Number of demo queries to run
        k: Number of results per query
    """
    import random
    
    print(f"🔍 Face Search Demo - {num_demos} random queries")
    print("=" * 60)
    
    # Get some random samples from the dataset
    available_indices = list(range(len(face_dataset)))
    demo_indices = random.sample(available_indices, min(num_demos, len(available_indices)))
    
    for i, idx in enumerate(demo_indices):
        _, label, query_path = face_dataset[idx]
        
        target_attr = face_dataset.target_attribute
        attr_value = "Yes" if label == 1 else "No"
        
        print(f"\n🎯 Demo {i+1}: {os.path.basename(query_path)}")
        print(f"   Target attribute '{target_attr}': {attr_value}")
        print("-" * 40)
        
        # Perform search and visualize
        retrieved_paths, distances, indices, query_attrs, retrieved_attrs = face_search_and_visualize(
            model, index, query_path, face_dataset, device, k=k
        )
        
        # Print retrieved results summary
        print("Retrieved results:")
        for j, (path, dist, ret_idx) in enumerate(zip(retrieved_paths, distances, indices)):
            retrieved_label = index.labels[ret_idx]
            retrieved_attr = "Yes" if retrieved_label == 1 else "No"
            match = "✅" if retrieved_label == label else "❌"
            print(f"  {j+1}. {os.path.basename(path)} | {target_attr}: {retrieved_attr} | Dist: {dist:.3f} {match}")


def analyze_face_search_performance(model, index, face_dataset: CelebADataset, 
                                   device: str, target_attribute: Optional[str] = None):
    """
    Analyze face search performance by attribute values.
    
    Args:
        model: ResNet embedder model
        index: FAISS search index
        face_dataset: CelebADataset instance
        device: Device for model inference
        target_attribute: Which attribute to analyze (defaults to dataset's target)
    """
    if target_attribute is None:
        target_attribute = face_dataset.target_attribute
    
    print(f"📊 Face Search Performance Analysis")
    print(f"Target Attribute: {target_attribute}")
    print("=" * 50)
    
    # Group queries by attribute value
    attr_0_queries = []  # e.g., Female faces
    attr_1_queries = []  # e.g., Male faces
    
    for i in range(min(20, len(face_dataset))):  # Analyze first 20 for speed
        _, label, path = face_dataset[i]
        if label == 0:
            attr_0_queries.append((i, path))
        else:
            attr_1_queries.append((i, path))
    
    # Analyze performance for each group
    for attr_value, queries, name in [(0, attr_0_queries, f"Non-{target_attribute}"), 
                                      (1, attr_1_queries, target_attribute)]:
        if not queries:
            continue
            
        print(f"\n🔍 {name} Faces Analysis:")
        
        precisions = []
        for i, (dataset_idx, query_path) in enumerate(queries[:5]):  # Test 5 per group
            # Search
            _, distances, indices, _, _ = face_search_and_visualize(
                model, index, query_path, face_dataset, device, k=5
            )
            
            # Calculate precision for this query
            relevant_count = sum(1 for idx in indices if index.labels[idx] == attr_value)
            precision = relevant_count / 5
            precisions.append(precision)
            
            print(f"   Query {i+1}: {os.path.basename(query_path)} - Precision@5: {precision:.2%}")
        
        avg_precision = np.mean(precisions) if precisions else 0
        print(f"   → Average Precision@5: {avg_precision:.2%}")


# Example usage function for face visualization
def setup_face_visualization_demo():
    """Setup function for face visualization demonstration."""
    print("🖼️ Face Search Visualization Setup")
    print("=" * 45)
    print("")
    print("Usage examples:")
    print("")
    print("1. Single query visualization:")
    print("   ```python")
    print("   results = face_search_and_visualize(")
    print("       model, face_index, query_path, face_dataset, device, k=5")
    print("   )")
    print("   ```")
    print("")
    print("2. Multiple demo queries:")
    print("   ```python")
    print("   demo_face_search(model, face_index, face_dataset, device, num_demos=3)")
    print("   ```")
    print("")
    print("3. Performance analysis:")
    print("   ```python")
    print("   analyze_face_search_performance(model, face_index, face_dataset, device)")
    print("   ```")
    print("")
    print("Features:")
    print("• Shows query face and top-k similar faces")
    print("• Displays key facial attributes (Male, Young, Smiling, etc.)")
    print("• Shows similarity distances")
    print("• Indicates correct/incorrect matches")
    print("• Performance analysis by attribute groups")


# Example usage function
def setup_face_search_demo():
    """Setup function for face search demonstration."""
    print("🔧 Face Search Extension Setup")
    print("=" * 40)
    print("")
    print("To use face search capabilities:")
    print("")
    print("1. Optional - Install face_recognition for alignment:")
    print("   pip install face_recognition")
    print("")
    print("2. Dataset structure (already detected):")
    print("   ./data/img_align_celeba/")
    print("   ├── img_align_celeba/            # Image files")
    print("   ├── list_attr_celeba.csv         # Attribute labels")
    print("   └── list_eval_partition.csv      # Train/val/test splits")
    print("")
    print("3. Example usage:")
    print("   ```python")
    print("   from face_search_extension import CelebADataset, get_face_transforms")
    print("   ")
    print("   # Create face dataset")
    print("   face_transform = get_face_transforms('val')")
    print("   face_dataset = CelebADataset('./data/img_align_celeba', 'val', face_transform)")
    print("   ")
    print("   # Use with existing search engine")
    print("   face_loader = DataLoader(face_dataset, batch_size=32, shuffle=False)")
    print("   face_embeddings, face_labels, face_paths = extract_embeddings(model, face_loader, device)")
    print("   face_index = FAISSIndex()")
    print("   face_index.build_index(face_embeddings, face_labels, face_paths)")
    print("   ")
    print("   # Evaluate with attribute metrics")
    print("   face_results = evaluate_face_search(model, face_index, face_dataset, device)")
    print("   ")
    print("   # NEW: Visualize search results")
    print("   demo_face_search(model, face_index, face_dataset, device)")
    print("   ```")
    print("")
    print("Available attributes for target_attribute parameter:")
    print("Male, Young, Smiling, Attractive, Blond_Hair, etc. (40 total)")


if __name__ == "__main__":
    setup_face_search_demo() 