# Facial Aging Pipeline with Stable Diffusion

A complete modular pipeline for realistic facial aging using Stable Diffusion inpainting, supporting both images and videos.

## Features

### Core Modules
- **Face Detection**: MediaPipe-based facial landmark detection and mask extraction
- **Stable Diffusion Inpainting**: HuggingFace integration for realistic aging effects
- **Enhanced Pipeline**: Advanced techniques for improved aging quality
- **Video Processing**: Frame-by-frame video aging with progress tracking

### Capabilities
- ✅ Single image facial aging
- ✅ Batch image processing
- ✅ Age series generation (multiple age increments)
- ✅ Video frame-by-frame processing
- ✅ Aging progression videos from single images
- ✅ Advanced prompting strategies
- ✅ Quality enhancement techniques
- ✅ Gradual multi-step aging for large age gaps

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

#### Single Image Aging
```python
from face_utils import get_face_mask
from inpaint_utils import inpaint_face

# Load and process image
mask = get_face_mask(image)
aged_image = inpaint_face(image, mask, "A realistic photo of the same person aged 15 years")
```

#### Video Processing
```python
from video_utils import process_video

# Process entire video
process_video(
    input_path="input_video.mp4",
    output_path="aged_video.mp4", 
    prompt="A realistic photo of the same person aged 15 years"
)
```

#### Enhanced Quality Pipeline
```python
from enhanced_pipeline import EnhancedFacialAgingPipeline

pipeline = EnhancedFacialAgingPipeline()
results = pipeline.process_age_series_enhanced(
    image_path="portrait.jpg",
    age_increments=[10, 20, 30],
    use_gradual=True
)
```

## Module Documentation

### 1. Face Utils (`face_utils.py`)

**Main Function:**
```python
def get_face_mask(image: np.ndarray, mask_type: str = "convex_hull") -> np.ndarray:
```

**Features:**
- MediaPipe facial landmark detection
- Two mask types: "convex_hull" (simple) and "facial_region" (precise)
- Binary mask output (255 inside face, 0 outside)
- Visualization utilities

**Example:**
```python
import cv2
from face_utils import get_face_mask, visualize_face_mask

image = cv2.imread("portrait.jpg")
mask = get_face_mask(image, mask_type="facial_region")
visualize_face_mask(image, mask)
```

### 2. Inpainting Utils (`inpaint_utils.py`)

**Main Function:**
```python
def inpaint_face(
    image: np.ndarray, 
    mask: np.ndarray, 
    prompt: str = "A realistic photo of the same person aged 10 years",
    negative_prompt: str = "blurry, distorted, unrealistic",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 0.85
) -> np.ndarray:
```

**Features:**
- HuggingFace Stable Diffusion integration
- Memory optimization and cleanup
- Batch processing capabilities
- Multiple age variant generation

**Example:**
```python
from inpaint_utils import inpaint_face, create_age_variants

# Single aging
aged_image = inpaint_face(image, mask, "Aged 20 years, graceful aging")

# Multiple age variants
variants = create_age_variants(image, mask, [5, 10, 15, 20])
```

### 3. Enhanced Pipeline (`enhanced_pipeline.py`)

**Main Class:**
```python
class EnhancedFacialAgingPipeline:
```

**Advanced Features:**
- Gradual multi-step aging for large age gaps
- High-quality parameter optimization
- Sophisticated prompt generation
- Technique comparison utilities

**Example:**
```python
from enhanced_pipeline import EnhancedFacialAgingPipeline

pipeline = EnhancedFacialAgingPipeline()

# Gradual aging for better quality
result = pipeline.process_gradual_aging(
    image_path="portrait.jpg",
    target_age=25,
    steps=3
)

# Compare different techniques
comparison = pipeline.compare_aging_techniques("portrait.jpg", age_years=15)
pipeline.visualize_comparison(comparison)
```

### 4. Video Processing (`video_utils.py`)

**Main Function:**
```python
def process_video(
    input_path: str, 
    output_path: str, 
    prompt: str = "A realistic photo of the same person aged 10 years"
) -> None:
```

**Video Features:**
- Frame-by-frame facial aging
- Automatic face detection per frame
- Pass-through for frames without faces
- Progress tracking with statistics
- Batch processing with multiple prompts
- Aging progression video creation

**Examples:**

```python
from video_utils import (
    process_video, 
    extract_video_info,
    create_aging_progression_video,
    process_video_batch
)

# Basic video processing
process_video("input.mp4", "aged_output.mp4", "Aged 15 years, natural aging")

# Get video information
info = extract_video_info("video.mp4")
print(f"Duration: {info['duration']:.1f}s, FPS: {info['fps']}")

# Create aging progression from single image
create_aging_progression_video(
    image_path="portrait.jpg",
    output_path="aging_progression.mp4",
    age_increments=[5, 10, 15, 20, 25],
    duration_per_age=3.0
)

# Batch processing with multiple prompts
prompts = [
    "Aged 10 years, natural progression",
    "Aged 15 years, graceful aging", 
    "Aged 20 years, distinguished appearance"
]
process_video_batch("input.mp4", "batch_output.mp4", prompts)
```

### 5. Advanced Prompting (`aging_prompts.py`)

**Features:**
- Age-specific characteristic prompts
- Gender-aware aging patterns
- Quality enhancement terms
- Negative prompt optimization

**Example:**
```python
from aging_prompts import get_premium_prompt, AgingPromptGenerator

# Get optimized prompts
positive, negative = get_premium_prompt(age_years=20)

# Generate custom prompts
generator = AgingPromptGenerator()
prompt = generator.generate_aging_prompt(
    age_years=15,
    gender_hint="feminine", 
    style="graceful"
)
```

## Example Notebooks

### 1. `example_face_mask.ipynb`
- Face detection and mask extraction demonstration
- Different mask types comparison
- Visualization utilities

### 2. `example_inpainting.ipynb` 
- Stable Diffusion inpainting examples
- Parameter tuning guidelines
- Quality optimization techniques

### 3. `complete_pipeline_example.ipynb`
- End-to-end pipeline demonstration
- Age series generation
- Advanced usage patterns

### 4. `video_processing_example.ipynb`
- Comprehensive video processing guide
- Frame-by-frame analysis
- Performance optimization tips
- Custom processing examples

## Performance Optimization

### Image Processing
- Use high-resolution images (512x512+) for best results
- Ensure faces are clearly visible and well-lit
- Higher inference steps (50+) improve quality
- Strength 0.8-0.9 preserves original features

### Video Processing
- Lower inference steps (20-30) for faster processing
- Process shorter clips first to test parameters
- Monitor GPU memory usage for long videos
- Consider parallel processing for large files

### Quality Tips
- Use "facial_region" mask type for precise aging
- Apply gradual aging for large age gaps (20+ years)
- Enhance prompts with quality modifiers
- Use negative prompts to avoid artifacts

## Hardware Requirements

### Minimum
- GPU: 6GB VRAM (GTX 1060 6GB / RTX 2060)
- RAM: 8GB system RAM
- Storage: 10GB free space

### Recommended  
- GPU: 8GB+ VRAM (RTX 3070 / RTX 4060 Ti)
- RAM: 16GB+ system RAM
- Storage: SSD for faster model loading

## Troubleshooting

### Common Issues

**1. Face Not Detected**
- Ensure face is clearly visible and front-facing
- Check image resolution and lighting
- Try different mask types

**2. Poor Aging Quality**
- Increase inference steps (50+)
- Use enhanced prompts from `aging_prompts.py`
- Try gradual aging for large age gaps
- Adjust strength parameter (0.8-0.9)

**3. GPU Memory Issues**
- Reduce batch size
- Lower inference steps for videos
- Enable gradient checkpointing
- Close other GPU applications

**4. Video Processing Slow**
- Use lower resolution videos for testing
- Reduce inference steps (20-30)
- Process shorter segments
- Consider cloud GPU services for large videos

## File Structure

```
notebooks/stable-diffusion/
├── face_utils.py                      # Face detection and masking
├── inpaint_utils.py                   # Stable Diffusion inpainting  
├── pipeline.py                        # Basic pipeline integration
├── enhanced_pipeline.py               # Advanced quality techniques
├── aging_prompts.py                   # Sophisticated prompting
├── video_utils.py                     # Video processing utilities
├── requirements.txt                   # Dependencies
├── __init__.py                        # Package initialization
├── example_face_mask.ipynb           # Face detection demo
├── example_inpainting.ipynb          # Inpainting examples
├── complete_pipeline_example.ipynb    # Complete pipeline demo
├── video_processing_example.ipynb     # Video processing guide
├── test_video_processing.py          # Video testing script
└── README.md                          # This documentation
```

## License and Credits

This pipeline combines several open-source technologies:
- **MediaPipe**: Google's face detection framework
- **Stable Diffusion**: RunwayML's inpainting model via HuggingFace
- **OpenCV**: Computer vision utilities
- **PyTorch**: Deep learning framework

## Contributing

To contribute:
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Submit pull request with documentation

## Future Enhancements

- [ ] Real-time video processing
- [ ] Style transfer integration
- [ ] Multi-face video support
- [ ] Mobile/edge deployment
- [ ] Custom model fine-tuning
- [ ] Web interface development 