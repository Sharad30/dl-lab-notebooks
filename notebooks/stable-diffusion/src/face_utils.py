"""
Face processing utilities for Stable Diffusion facial aging pipeline.

This module provides utilities for:
- Extracting facial masks using MediaPipe
- Processing facial landmarks
- Generating binary masks for inpainting
"""

import numpy as np
import cv2
try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed. Run: pip install mediapipe")
    mp = None

from typing import Optional, Tuple, List, Union
import warnings

# Suppress MediaPipe warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


class FaceMaskExtractor:
    """Extract facial masks using MediaPipe face detection and landmarks."""
    
    def __init__(self):
        """Initialize MediaPipe face detection and face mesh models."""
        if mp is None:
            raise ImportError("MediaPipe is required but not installed")
            
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def get_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from image using MediaPipe.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Array of facial landmarks as (x, y) coordinates, or None if no face detected
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to pixel coordinates
        h, w = image.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
            
        return np.array(landmarks, dtype=np.int32)
    
    def create_convex_hull_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create a binary mask using convex hull of facial landmarks.
        
        Args:
            landmarks: Facial landmarks as (x, y) coordinates
            image_shape: Shape of the target image
            
        Returns:
            Binary mask with 255 inside face, 0 outside
        """
        # Create empty mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Compute convex hull of all landmarks
        hull = cv2.convexHull(landmarks)
        
        # Fill the convex hull area (fillPoly expects a list of contours and a scalar value)
        cv2.fillPoly(mask, [hull], (255,))
        
        return mask
    
    def create_facial_region_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create a more precise facial mask using specific facial regions.
        
        Args:
            landmarks: Facial landmarks as (x, y) coordinates  
            image_shape: Shape of the target image
            
        Returns:
            Binary mask with 255 inside face, 0 outside
        """
        # Create empty mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # MediaPipe face mesh landmark indices for facial outline
        # Face oval indices (approximate face boundary)
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Extract face oval points
        if len(landmarks) >= max(face_oval_indices) + 1:
            face_oval_points = landmarks[face_oval_indices]
        else:
            # Fallback to convex hull if we don't have enough landmarks
            face_oval_points = cv2.convexHull(landmarks)
            
        # Fill the facial region (fillPoly expects a list of contours and a scalar value)
        cv2.fillPoly(mask, [face_oval_points], (255,))
        
        return mask


def get_face_mask(image: np.ndarray, mask_type: str = "convex_hull") -> Optional[np.ndarray]:
    """
    Extract a binary face mask from an image using MediaPipe.
    
    Args:
        image: Input image as numpy array (BGR or RGB format)
        mask_type: Type of mask to create ("convex_hull" or "facial_region")
        
    Returns:
        Binary mask (np.uint8) with 255 inside face, 0 outside.
        Returns None if no face is detected.
        
    Example:
        >>> import cv2
        >>> image = cv2.imread("face.jpg")
        >>> mask = get_face_mask(image)
        >>> if mask is not None:
        ...     cv2.imwrite("face_mask.jpg", mask)
    """
    # Initialize face mask extractor
    extractor = FaceMaskExtractor()
    
    # Get facial landmarks
    landmarks = extractor.get_face_landmarks(image)
    
    if landmarks is None:
        print("No face detected in the image")
        return None
    
    # Create mask based on specified type
    if mask_type == "convex_hull":
        mask = extractor.create_convex_hull_mask(landmarks, image.shape)
    elif mask_type == "facial_region":
        mask = extractor.create_facial_region_mask(landmarks, image.shape)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. Use 'convex_hull' or 'facial_region'")
    
    return mask


def visualize_face_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Visualize the face mask overlaid on the original image.
    
    Args:
        image: Original image
        mask: Binary face mask
        alpha: Transparency of the mask overlay (0.0 to 1.0)
        
    Returns:
        Image with mask overlay for visualization
    """
    # Convert mask to 3-channel
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply color to the mask (red color)
    mask_colored[:, :, 0] = 0  # Blue channel
    mask_colored[:, :, 1] = 0  # Green channel
    # Red channel remains as mask values
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    
    return result


# Convenience function for quick testing
def test_face_mask_extraction(image_path: str, output_path: Optional[str] = None):
    """
    Test the face mask extraction on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save the result (optional)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Extract face mask
    mask = get_face_mask(image)
    
    if mask is None:
        print("No face detected in the image")
        return
    
    # Create visualization
    visualization = visualize_face_mask(image, mask)
    
    if output_path:
        # Save results
        cv2.imwrite(output_path.replace('.', '_mask.'), mask)
        cv2.imwrite(output_path.replace('.', '_visualization.'), visualization)
        print(f"Results saved to {output_path}")
    else:
        # Display results (if in interactive environment)
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Face Mask")
            axes[1].axis('off')
            
            axes[2].imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Mask Overlay")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization") 