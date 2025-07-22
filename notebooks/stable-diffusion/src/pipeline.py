"""
Complete facial aging pipeline class.

This module provides a high-level interface for the facial aging pipeline,
combining face detection and Stable Diffusion inpainting.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Union
from face_utils import get_face_mask, visualize_face_mask
from inpaint_utils import inpaint_face, create_age_variants


class FacialAgingPipeline:
    """
    Complete facial aging pipeline combining face detection and Stable Diffusion inpainting.
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        """
        Initialize the facial aging pipeline.
        
        Args:
            model_id: Stable Diffusion model to use
        """
        self.model_id = model_id
        print(f"Initialized Facial Aging Pipeline with model: {model_id}")
    
    def process_single_image(
        self, 
        image_path: str, 
        age_years: int = 10,
        mask_type: str = "convex_hull",
        **kwargs
    ) -> Dict[str, Union[np.ndarray, int]]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            age_years: Years to age the person
            mask_type: Type of face mask ("convex_hull" or "facial_region")
            **kwargs: Additional arguments for inpainting
            
        Returns:
            Dictionary with original image, mask, and aged result
        """
        print(f"Processing: {image_path}")
        
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 2: Extract face mask
        print("  → Extracting face mask...")
        mask = get_face_mask(image, mask_type=mask_type)
        if mask is None:
            raise ValueError("No face detected in image")
        
        # Step 3: Perform aging
        print(f"  → Aging face by {age_years} years...")
        prompt = f"A realistic photo of the same person aged {age_years} years"
        
        aged_image = inpaint_face(
            image=image,
            mask=mask,
            prompt=prompt,
            model_id=self.model_id,
            **kwargs
        )
        
        print("  ✅ Processing complete!")
        
        return {
            "original": image,
            "mask": mask,
            "aged": aged_image,
            "age_years": age_years
        }
    
    def process_age_series(
        self, 
        image_path: str, 
        age_increments: List[int] = [5, 10, 15, 20],
        **kwargs
    ) -> Dict[str, Union[np.ndarray, int]]:
        """
        Create a series of aged versions of the same face.
        
        Args:
            image_path: Path to input image
            age_increments: List of years to age
            **kwargs: Additional arguments for inpainting
            
        Returns:
            Dictionary with original image, mask, and all aged variants
        """
        print(f"Creating age series for: {image_path}")
        
        # Load and prepare
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        mask = get_face_mask(image)
        if mask is None:
            raise ValueError("No face detected in image")
        
        # Create age variants
        print(f"  → Creating {len(age_increments)} age variants...")
        age_variants = create_age_variants(
            image=image,
            mask=mask,
            age_increments=age_increments,
            model_id=self.model_id,
            **kwargs
        )
        
        result = {
            "original": image,
            "mask": mask,
            **age_variants
        }
        
        print("  ✅ Age series complete!")
        return result
    
    def visualize_results(self, results: Dict, title: str = "Facial Aging Results"):
        """
        Visualize pipeline results.
        
        Args:
            results: Results dictionary from processing
            title: Plot title
        """
        # Count non-metadata items
        plot_items = {k: v for k, v in results.items() 
                     if k not in ['mask', 'age_years'] and isinstance(v, np.ndarray)}
        
        num_items = len(plot_items)
        if num_items == 0:
            print("No images to display")
            return
        
        # Create subplot layout
        cols = min(5, num_items)
        rows = (num_items + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Convert axes to a flat list for consistent indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        elif cols == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot images
        for idx, (key, image) in enumerate(plot_items.items()):
            axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if key == "original":
                axes[idx].set_title("Original")
            elif isinstance(key, int):  # Handle integer age keys
                axes[idx].set_title(f"Aged +{key} Years")
            elif isinstance(key, str) and key.isdigit():  # Handle string digit keys
                axes[idx].set_title(f"Aged +{key} Years")
            else:
                axes[idx].set_title(str(key).replace("_", " ").title())
            
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(plot_items), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def save_results(self, results: Dict, output_dir: str, base_name: str = "result"):
        """
        Save results to disk.
        
        Args:
            results: Results dictionary
            output_dir: Directory to save to
            base_name: Base filename
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for key, image in results.items():
            if isinstance(image, np.ndarray) and key != 'mask':
                if key == "original":
                    filename = f"{base_name}_original.jpg"
                elif key.isdigit():
                    filename = f"{base_name}_aged_{key}years.jpg"
                else:
                    filename = f"{base_name}_{key}.jpg"
                
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, image)
                saved_files.append(filename)
        
        # Save mask separately
        if 'mask' in results:
            mask_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
            cv2.imwrite(mask_path, results['mask'])
            saved_files.append(f"{base_name}_mask.jpg")
        
        print(f"Saved {len(saved_files)} files to {output_dir}:")
        for filename in saved_files:
            print(f"  → {filename}") 