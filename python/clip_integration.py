"""
CLIP Model Wrapper
Handles loading and interfacing with CLIP ViT-B/32 for UAP generation

This module provides a clean interface for:
1. Loading CLIP model weights (ViT-B/32)
2. Preprocessing images consistently
3. Extracting semantic embeddings
4. Computing image-text similarities
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import Union, List, Tuple
import os


class CLIPModelWrapper:
    """
    Clean wrapper for OpenAI's CLIP (ViT-B/32) model
    Ensures consistent preprocessing and embedding extraction
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant (default: "ViT-B/32")
            device: Device to run model on. If None, auto-detects GPU
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[CLIP] Loading {model_name} on {self.device}...")
        
        # Load model and preprocessing transform
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"[CLIP] Model loaded successfully")
        print(f"[CLIP] Parameters: {num_params:,}")
        print(f"[CLIP] Image size: 224x224")
        print(f"[CLIP] Embedding dimension: 512")
    
    def get_embeddings(self, image_path: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract semantic embedding vector from an image
        This converts an image into a 512-dimensional semantic representation
        
        Args:
            image_path: Path to image file, PIL Image, or numpy array
            
        Returns:
            Normalized embedding vector (512-dimensional numpy array)
        """
        # Load and preprocess image
        image_tensor = self._load_image(image_path)
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # L2 normalize (standard for CLIP)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy().squeeze()
        
        return embedding
    
    def get_text_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract semantic embedding from text description(s)
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Normalized embedding vector(s) (512-dimensional)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        text_tokens = clip.tokenize(text).to(self.device)
        
        # Extract features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = text_features.cpu().numpy()
        
        return embeddings.squeeze() if len(text) == 1 else embeddings
    
    def compute_similarity(self, image_path: Union[str, np.ndarray], 
                          text: str) -> float:
        """
        Compute cosine similarity between image and text
        
        Args:
            image_path: Path to image or image data
            text: Text description
            
        Returns:
            Similarity score (0-100, higher = more similar)
        """
        # Get embeddings
        image_emb = self.get_embeddings(image_path)
        text_emb = self.get_text_embeddings(text)
        
        # Compute cosine similarity (already normalized, so just dot product)
        similarity = np.dot(image_emb, text_emb) * 100.0
        
        return float(similarity)
    
    def batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract embeddings for multiple images efficiently
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Stacked embeddings (N x 512 numpy array)
        """
        embeddings = []
        
        for img_path in image_paths:
            emb = self.get_embeddings(img_path)
            embeddings.append(emb)
        
        return np.stack(embeddings)
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Internal method to load and preprocess image
        Ensures consistent preprocessing across all operations
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image tensor (1, 3, 224, 224)
        """
        # Convert to PIL Image if needed
        if isinstance(image_input, str):
            # Load from file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL
            if image_input.max() <= 1.0:
                image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input.astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            # Already PIL Image
            image = image_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def get_model(self) -> torch.nn.Module:
        """
        Get direct access to CLIP model for advanced operations
        
        Returns:
            CLIP model instance
        """
        return self.model
    
    def get_preprocess(self):
        """
        Get CLIP preprocessing function
        
        Returns:
            Preprocessing transform
        """
        return self.preprocess


def demo():
    """
    Demonstration of CLIP model wrapper usage
    """
    print("="*60)
    print("CLIP Model Wrapper Demo")
    print("="*60)
    
    # Initialize wrapper
    clip_model = CLIPModelWrapper(model_name="ViT-B/32")
    
    # Example 1: Get image embeddings
    print("\n--- Example 1: Image Embeddings ---")
    test_img = os.path.join('..', 'data', 'test_img.png')
    if os.path.exists(test_img):
        embedding = clip_model.get_embeddings(test_img)
        print(f"Image: {test_img}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
        print(f"Embedding sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
    else:
        print(f"Test image not found: {test_img}")
    
    # Example 2: Get text embeddings
    print("\n--- Example 2: Text Embeddings ---")
    text = "a photo of a cat"
    text_emb = clip_model.get_text_embeddings(text)
    print(f"Text: '{text}'")
    print(f"Embedding shape: {text_emb.shape}")
    print(f"Embedding norm: {np.linalg.norm(text_emb):.4f}")
    
    # Example 3: Compute similarity
    if os.path.exists(test_img):
        print("\n--- Example 3: Image-Text Similarity ---")
        texts = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a person",
            "a photo of nature"
        ]
        
        for text in texts:
            similarity = clip_model.compute_similarity(test_img, text)
            print(f"'{text:30s}': {similarity:6.2f}%")
    
    # Example 4: Batch processing
    print("\n--- Example 4: Batch Embeddings ---")
    import glob
    coco_images = glob.glob(os.path.join('..', '..', '*.jpg'))[:5]
    if len(coco_images) > 0:
        print(f"Processing {len(coco_images)} images...")
        batch_embs = clip_model.batch_embeddings(coco_images)
        print(f"Batch embeddings shape: {batch_embs.shape}")
        print(f"Average norm: {np.mean(np.linalg.norm(batch_embs, axis=1)):.4f}")
    else:
        print("No MS-COCO images found for batch demo")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()
