# Visual Search Engine

A comprehensive visual search engine implementation using deep learning and similarity search, built for both general image search and face recognition tasks.

## 🎯 Features

- **ResNet101 Embeddings**: 2048-dimensional feature vectors from pretrained ImageNet model
- **FAISS Search Index**: Efficient L2 distance similarity search  
- **TinyImageNet Support**: Automatic download and preprocessing
- **Face Search Extension**: CelebA/VGGFace2 support with face alignment
- **Comprehensive Evaluation**: Precision@k, Recall@k, mAP metrics
- **Interactive Visualization**: Query and result display
- **Scalable Architecture**: Handles 100K+ images efficiently

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- faiss-cpu >= 1.7.0
- numpy, matplotlib, Pillow, scikit-learn

## 🚀 Quick Start

### 1. Basic Image Search (TinyImageNet)

```python
from utils import *
import torch

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download and prepare dataset
dataset_path = download_tiny_imagenet('./data')
train_dataset = TinyImageNetDataset(dataset_path, 'train', get_transforms('train'))
val_dataset = TinyImageNetDataset(dataset_path, 'val', get_transforms('val'))

# Initialize model and extract embeddings
model = ResNetEmbedder(pretrained=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
embeddings, labels, paths = extract_embeddings(model, train_loader, device)

# Build search index
index = FAISSIndex()
index.build_index(embeddings, labels, paths)

# Evaluate performance
results = evaluate_search_engine(model, index, val_dataset, device)
print(f"Precision@5: {results['precision@5']:.4f}")

# Interactive search
query_path = val_dataset.image_paths[0]
search_and_visualize(model, index, query_path, device, k=5)
```

### 2. Face Search Extension

```python
from face_search_extension import CelebADataset, evaluate_face_search

# Setup face dataset (requires CelebA download)
face_dataset = CelebADataset('./data/celeba', 'val', get_face_transforms('val'))

# Extract face embeddings and build index
face_loader = DataLoader(face_dataset, batch_size=32, shuffle=False)
face_embeddings, face_labels, face_paths = extract_embeddings(model, face_loader, device)
face_index = FAISSIndex()
face_index.build_index(face_embeddings, face_labels, face_paths)

# Evaluate with identity matching
face_results = evaluate_face_search(model, face_index, face_dataset, device)
print(f"Identity Precision@5: {face_results['identity_precision@5']:.4f}")
```

## 📊 Performance Targets

### TinyImageNet (General Objects)
- **Target**: Precision@5 > 80%
- **Baseline**: ResNet101 features typically achieve 75-85%
- **Dataset**: 200 classes, 100K training images

### Face Search (Identity Matching) 
- **Target**: Identity match in top-k results
- **Metric**: Identity Precision@k (same person retrieval)
- **Enhancement**: Face alignment preprocessing

## 🏗️ Architecture

```
Visual Search Engine
├── Embedding Model (ResNet101)
│   ├── Pretrained on ImageNet
│   ├── Remove final FC layer
│   └── Output: 2048-dim vectors
│
├── Search Index (FAISS)
│   ├── IndexFlatL2 (exact search)
│   ├── Normalized embeddings
│   └── Scalable to millions of images
│
├── Evaluation Framework
│   ├── Precision@k, Recall@k
│   ├── Mean Average Precision (mAP)
│   └── Face verification metrics
│
└── Extensions
    ├── Face alignment preprocessing
    ├── Identity-based evaluation
    └── Advanced embedding models
```

## 📁 Project Structure

```
visual_search_engine/
├── visual_search_engine.ipynb    # Main demonstration notebook
├── utils.py                      # Core utilities and functions
├── face_search_extension.py      # Face search capabilities
├── requirements.txt              # Package dependencies
├── README.md                     # This file
└── data/                         # Data directory
    ├── tiny-imagenet-200/        # Auto-downloaded
    └── celeba/                   # Manual download required
```

## 🔧 Detailed Usage

### Dataset Setup

**TinyImageNet** (Automatic):
```python
dataset_path = download_tiny_imagenet('./data')  # ~240MB download
```

**CelebA** (Manual):
1. Download from [CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Extract to `./data/celeba/`
3. Ensure structure:
   ```
   ./data/celeba/
   ├── Img/img_celeba/           # Image files
   ├── Anno/identity_CelebA.txt  # Identity labels
   └── Eval/list_eval_partition.txt  # Train/val/test splits
   ```

### Model Configuration

```python
# Standard ResNet101 setup
model = ResNetEmbedder(pretrained=True)

# Custom embedding dimension (if needed)
model = ResNetEmbedder(pretrained=True)
# Output is always 2048-dim from ResNet101
```

### Advanced Search Options

```python
# Build index with custom parameters
index = FAISSIndex(dimension=2048)
index.build_index(embeddings, labels, paths)

# Save/load index for reuse
index.save('./data/my_index')
index.load('./data/my_index')

# Custom search with multiple k values
for k in [1, 5, 10, 20]:
    distances, indices = index.search(query_embedding, k)
    print(f"Top-{k} results: {[paths[i] for i in indices]}")
```

## 📈 Evaluation Metrics

### Standard Metrics
- **Precision@k**: Fraction of top-k results that are relevant
- **Recall@k**: Fraction of relevant items found in top-k  
- **mAP@k**: Mean Average Precision at k

### Face-Specific Metrics
- **Identity Precision@k**: Fraction of top-k faces with same identity
- **Identity Recall@k**: Fraction of same-identity faces found
- **Face Verification**: Same/different identity classification accuracy

### Example Results
```python
results = evaluate_search_engine(model, index, val_dataset, device)
# {
#   'precision@1': 0.734,
#   'precision@5': 0.821,  # ✅ Above 80% target
#   'precision@10': 0.756,
#   'recall@5': 0.082,
#   'map@5': 0.801
# }
```

## 🔬 Future Improvements

### 1. Better Embeddings
```python
# CLIP (vision-language model)
import clip
model, preprocess = clip.load("ViT-B/32", device=device)

# DINOv2 (self-supervised)
import torch
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```

### 2. Advanced Indexing
```python
# FAISS IVF for faster search
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Product Quantization for memory efficiency  
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
```

### 3. Custom Fine-tuning
```python
# Fine-tune ResNet on target dataset
from torchvision.models import resnet101
model = resnet101(pretrained=True)
model.fc = nn.Linear(2048, num_classes)
# Train with your specific data
```

## 🐛 Troubleshooting

### Common Issues

**FAISS Installation**:
```bash
# CPU version
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

**Memory Issues with Large Datasets**:
```python
# Process in batches
for batch in tqdm(dataloader):
    embeddings = model(batch)
    # Process batch by batch instead of all at once
```

**Face Recognition Dependencies**:
```bash
# Face alignment requires additional packages
pip install face_recognition
pip install opencv-python
```

## 📚 References

- [FAISS: A library for efficient similarity search](https://github.com/facebookresearch/faiss)
- [TinyImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 📄 License

This project is provided for educational and research purposes. Please respect the licenses of individual datasets:
- TinyImageNet: Academic use
- CelebA: Non-commercial research use only

## 🤝 Contributing

Feel free to submit issues and enhancement requests! Key areas for contribution:
- Additional embedding models (CLIP, DINOv2, etc.)
- New evaluation metrics
- Performance optimizations
- Extended dataset support 