"""
CLIP Universal Adversarial Perturbation Generator
Core algorithm for generating semantic UAPs against CLIP ViT-B/32

This is the MAIN algorithm that creates the "Universal Cloak" by:
1. Minimizing cosine similarity between images and text descriptions
2. Using alpha blending for mobile app compatibility
3. Memory-efficient mini-batch processing
"""

import numpy as np
import torch
import os
from tqdm import tqdm
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

from clip_integration import CLIPModelWrapper
from coco_loader import COCODataLoader


class UniversalPerturbationGenerator:
    """
    Generates Universal Adversarial Perturbations for CLIP model
    using semantic loss optimization with alpha blending
    """
    
    def __init__(self, 
                 clip_model: CLIPModelWrapper,
                 data_loader: COCODataLoader,
                 device: str = None):
        """
        Initialize UAP generator
        
        Args:
            clip_model: Initialized CLIP model wrapper
            data_loader: COCO data loader
            device: Torch device
        """
        self.clip_model = clip_model
        self.data_loader = data_loader
        self.device = device if device else clip_model.device
        
        # Get model for gradient computation
        self.model = clip_model.get_model()
        self.preprocess = clip_model.get_preprocess()
        
        print("[UAP] Generator initialized")
    
    def _semantic_loss(self, image_tensor: torch.Tensor, 
                       text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic loss: cosine similarity between image and text
        
        Goal: MINIMIZE this loss to "cloak" the image from AI understanding
        Lower similarity = better protection
        
        Args:
            image_tensor: Batch of images (B, 3, 224, 224)
            text_features: Normalized text embeddings (B, 512)
            
        Returns:
            Mean cosine similarity (scalar tensor)
        """
        # Encode images
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        # Shape: (B, B) but we only need diagonal (paired similarities)
        similarity = (image_features * text_features).sum(dim=1)
        
        # Return mean similarity as loss
        return similarity.mean()
    
    def _apply_alpha_blending(self, 
                             original: torch.Tensor, 
                             perturbation: torch.Tensor,
                             alpha: float) -> torch.Tensor:
        """
        Apply alpha blending: Image_new = (1 - α) · Image + α · (Image + UAP)
        
        This ensures UAP is optimized for the EXACT blending used in mobile app
        
        Args:
            original: Original image tensor
            perturbation: UAP tensor
            alpha: Blending factor (0 to 1)
            
        Returns:
            Blended image tensor
        """
        # Add perturbation to original
        perturbed = torch.clamp(original + perturbation, 0, 1)
        
        # Alpha blend
        blended = (1 - alpha) * original + alpha * perturbed
        blended = torch.clamp(blended, 0, 1)
        
        return blended
    
    def _project_perturbation(self, 
                             perturbation: torch.Tensor,
                             xi: float,
                             norm_type: str = "inf") -> torch.Tensor:
        """
        Project perturbation onto L_p ball
        Ensures perturbation stays within bounded magnitude
        
        Args:
            perturbation: Current perturbation
            xi: Ball radius (max magnitude)
            norm_type: "inf" or "2"
            
        Returns:
            Projected perturbation
        """
        if norm_type == "inf":
            # L-infinity: element-wise clipping
            return torch.clamp(perturbation, -xi, xi)
        elif norm_type == "2":
            # L2: scale if norm exceeds xi
            norm = torch.norm(perturbation)
            if norm > xi:
                return perturbation * (xi / norm)
            return perturbation
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def generate(self,
                 batch_size: int = 1000,
                 num_iterations: int = 10,
                 xi: float = 16/255,
                 learning_rate: float = 2/255,
                 alpha: float = 0.7,
                 delta: float = 0.2,
                 norm_type: str = "inf",
                 save_checkpoints: bool = True,
                 checkpoint_dir: str = "../data",
                 seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate Universal Adversarial Perturbation
        
        KEY ALGORITHM - This is where the magic happens!
        
        Args:
            batch_size: Number of images per mini-batch (1000 for CPU efficiency)
            num_iterations: Number of optimization passes
            xi: Perturbation magnitude bound (16/255 ≈ 0.063)
            learning_rate: Gradient descent step size
            alpha: Alpha blending for mobile app (0.7 = 70% blend)
            delta: Target fooling rate (0.2 = 80% fooling)
            norm_type: "inf" (recommended) or "2"
            save_checkpoints: Save perturbation after each iteration
            checkpoint_dir: Directory for checkpoints
            seed: Random seed for reproducibility
            
        Returns:
            Universal perturbation tensor (1, 3, 224, 224)
        """
        
        print("="*60)
        print("UNIVERSAL ADVERSARIAL PERTURBATION GENERATION")
        print("Algorithm: Semantic Loss Minimization for CLIP")
        print("="*60)
        print(f"Dataset size:       {len(self.data_loader):,} images")
        print(f"Mini-batch size:    {batch_size:,} images")
        print(f"Iterations:         {num_iterations}")
        print(f"Perturbation bound: {xi:.4f} (L-{norm_type})")
        print(f"Learning rate:      {learning_rate:.4f}")
        print(f"Alpha blending:     {alpha:.2f} (mobile app)")
        print(f"Target fooling:     {(1-delta)*100:.1f}%")
        print("="*60)
        
        # Initialize perturbation
        v = torch.zeros(1, 3, 224, 224, device=self.device)
        
        # Track metrics
        history = {
            'iterations': [],
            'losses': [],
            'avg_similarities': [],
            'fooling_rates': []
        }
        
        # Compute baseline
        print("\n[Phase 1] Computing baseline similarities...")
        baseline_batch = self.data_loader.get_mini_batch(min(batch_size, 500), seed=seed)
        baseline_sim = self._compute_baseline(baseline_batch[:100])
        print(f"Baseline avg similarity: {baseline_sim:.2f}%")
        
        # Main optimization loop
        print(f"\n[Phase 2] Optimizing perturbation...")
        print("-"*60)
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
            
            # Get mini-batch
            image_paths = self.data_loader.get_mini_batch(batch_size, seed=seed)
            text_descriptions = self.data_loader.create_text_descriptions_for_batch(batch_size)
            
            # Pre-compute text features
            text_tokens = torch.cat([
                torch.tensor(self.clip_model.model.encode_text(
                    torch.tensor([desc]).to(self.device)
                )).to(self.device) 
                for desc in text_descriptions
            ])
            
            # Load and process images
            print("Loading mini-batch...")
            dataset = []
            for img_path in tqdm(image_paths[:batch_size], desc="Loading", ncols=80):
                try:
                    img_tensor = self.clip_model._load_image(img_path)
                    dataset.append(img_tensor)
                except Exception as e:
                    continue
            
            if len(dataset) == 0:
                print("Warning: No valid images in batch, skipping...")
                continue
            
            # Pre-compute text features properly
            print("Pre-computing text embeddings...")
            text_features_list = []
            for desc in tqdm(text_descriptions[:len(dataset)], desc="Text", ncols=80):
                text_emb = self.clip_model.get_text_embeddings(desc)
                text_features_list.append(torch.from_numpy(text_emb).to(self.device))
            text_features = torch.stack(text_features_list)
            
            # Optimization loop over batch
            epoch_loss = 0.0
            num_processed = 0
            
            print("Optimizing...")
            for idx in tqdm(range(len(dataset)), desc="Optimize", ncols=80):
                img = dataset[idx]
                text_feat = text_features[idx:idx+1]
                
                # Create perturbation copy with gradient tracking
                v_grad = v.clone().detach().requires_grad_(True)
                
                # Apply alpha blending
                blended = self._apply_alpha_blending(img, v_grad, alpha)
                
                # Compute semantic loss
                loss = self._semantic_loss(blended, text_feat)
                
                # Backward pass
                loss.backward()
                
                # Update perturbation (gradient ascent to maximize loss = minimize similarity)
                with torch.no_grad():
                    # FGSM-style update: move in direction that increases loss
                    grad = v_grad.grad
                    v = v + learning_rate * grad.sign()
                    
                    # Project onto L_p ball
                    v = self._project_perturbation(v, xi, norm_type)
                
                epoch_loss += loss.item()
                num_processed += 1
            
            avg_loss = epoch_loss / num_processed if num_processed > 0 else 0
            
            # Evaluate
            print("\nEvaluating perturbation...")
            eval_results = self._evaluate(dataset[:100], text_features[:100], v, alpha, baseline_sim)
            
            # Store metrics
            history['iterations'].append(iteration + 1)
            history['losses'].append(avg_loss)
            history['avg_similarities'].append(eval_results['avg_similarity'])
            history['fooling_rates'].append(eval_results['fooling_rate'])
            
            # Print results
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1} Complete")
            print(f"{'='*60}")
            print(f"Avg Loss:           {avg_loss:.4f}")
            print(f"Original Sim:       {baseline_sim:.2f}%")
            print(f"Protected Sim:      {eval_results['avg_similarity']:.2f}%")
            print(f"Similarity Drop:    {baseline_sim - eval_results['avg_similarity']:.2f}%")
            print(f"Fooling Rate:       {eval_results['fooling_rate']*100:.2f}%")
            print(f"Target:             {(1-delta)*100:.1f}%")
            print(f"Perturbation L-inf: {torch.max(torch.abs(v)).item():.6f}")
            print(f"{'='*60}")
            
            # Save checkpoint
            if save_checkpoints:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"uap_checkpoint_iter{iteration+1}.npy"
                )
                np.save(checkpoint_path, v.cpu().numpy())
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Check if target reached
            if eval_results['fooling_rate'] >= (1 - delta):
                print(f"\n✓ Target fooling rate achieved!")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("UAP GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Final iteration:    {iteration + 1}")
        print(f"Final fooling rate: {eval_results['fooling_rate']*100:.2f}%")
        print(f"Similarity drop:    {baseline_sim - eval_results['avg_similarity']:.2f}%")
        print(f"Perturbation norm:  {torch.max(torch.abs(v)).item():.6f}")
        print(f"{'='*60}")
        
        # Save final perturbation
        final_path = os.path.join(checkpoint_dir, "clip_uap_final.npy")
        np.save(final_path, v.cpu().numpy())
        print(f"\n✓ Saved final UAP: {final_path}")
        
        # Save training history
        self._save_history(history, checkpoint_dir)
        
        return v
    
    def _compute_baseline(self, image_paths: List[str]) -> float:
        """Compute average baseline similarity"""
        similarities = []
        
        for img_path in image_paths[:100]:
            try:
                sim = self.clip_model.compute_similarity(img_path, "a photograph")
                similarities.append(sim)
            except:
                continue
        
        return np.mean(similarities) if similarities else 50.0
    
    def _evaluate(self, 
                  dataset: List[torch.Tensor],
                  text_features: torch.Tensor,
                  perturbation: torch.Tensor,
                  alpha: float,
                  baseline_sim: float) -> Dict:
        """Evaluate perturbation on dataset"""
        
        similarities = []
        fooled_count = 0
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                img = dataset[idx]
                text_feat = text_features[idx:idx+1]
                
                # Apply perturbation
                blended = self._apply_alpha_blending(img, perturbation, alpha)
                
                # Compute similarity
                img_feat = self.model.encode_image(blended)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * (img_feat * text_feat).sum()).item()
                
                similarities.append(sim)
                
                # Count as fooled if similarity dropped >30%
                if sim < baseline_sim * 0.7:
                    fooled_count += 1
        
        return {
            'avg_similarity': np.mean(similarities),
            'fooling_rate': fooled_count / len(dataset)
        }
    
    def _save_history(self, history: Dict, save_dir: str):
        """Save training history and plot"""
        
        # Save raw data
        history_path = os.path.join(save_dir, "training_history.npy")
        np.save(history_path, history)
        print(f"✓ Saved training history: {history_path}")
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history['iterations'], history['losses'], 'b-o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Similarity
        axes[0, 1].plot(history['iterations'], history['avg_similarities'], 'r-o')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Avg Similarity (%)')
        axes[0, 1].set_title('Protected Image Similarity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fooling rate
        axes[1, 0].plot(history['iterations'], 
                       [r*100 for r in history['fooling_rates']], 'g-o')
        axes[1, 0].axhline(y=80, color='r', linestyle='--', label='Target 80%')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Fooling Rate (%)')
        axes[1, 0].set_title('Fooling Rate Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        
        Total Iterations: {len(history['iterations'])}
        
        Final Metrics:
        - Loss: {history['losses'][-1]:.4f}
        - Similarity: {history['avg_similarities'][-1]:.2f}%
        - Fooling Rate: {history['fooling_rates'][-1]*100:.2f}%
        
        Best Fooling Rate: {max(history['fooling_rates'])*100:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Saved training plot: {plot_path}")
        plt.close()


def main():
    """
    Main entry point for UAP generation
    """
    print("Initializing UAP Generator...")
    print("-"*60)
    
    # Initialize components
    print("\n[1/3] Loading CLIP model...")
    clip_model = CLIPModelWrapper(model_name="ViT-B/32")
    
    print("\n[2/3] Loading MS-COCO dataset...")
    data_loader = COCODataLoader()
    
    print("\n[3/3] Initializing generator...")
    generator = UniversalPerturbationGenerator(
        clip_model=clip_model,
        data_loader=data_loader
    )
    
    # Generate UAP
    print("\n" + "="*60)
    print("Starting UAP Generation")
    print("="*60)
    
    perturbation = generator.generate(
        batch_size=1000,      # Memory-efficient batching
        num_iterations=10,    # 10 passes over data
        xi=16/255,           # Perturbation bound
        learning_rate=2/255, # Step size
        alpha=0.7,           # Alpha blending for mobile
        delta=0.2,           # 80% fooling target
        norm_type="inf",     # L-infinity norm
        save_checkpoints=True,
        seed=42              # Reproducibility
    )
    
    print("\n✓ UAP generation complete!")
    print("Next steps:")
    print("  1. Run fidelity_validator.py to check SSIM/PSNR")
    print("  2. Test on new images")
    print("  3. Deploy to mobile app")


if __name__ == "__main__":
    main()
