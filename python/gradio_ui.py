"""
Gradio Web UI for VeilAI - Image Protection System
Provides interactive interface for image cloaking and protection

Features:
- Image upload and visualization
- UAP generation and application
- Fidelity validation (SSIM, PSNR)
- Mobile-optimized alpha blending preview
- Batch processing
- Model management
"""

import gradio as gr
import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO

# Handle optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed - some features will be unavailable")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ scikit-image not installed - metrics unavailable")

# Import project modules with graceful fallback
try:
    from clip_integration import CLIPModelWrapper
    from coco_loader import COCODataLoader
    from clip_uap_generator import UniversalPerturbationGenerator
    from fidelity_validator import FidelityValidator
    from visualize_cloak import visualize_mobile_cloak
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"⚠️ Some project modules unavailable: {str(e)}")


# ============================================================================
# GLOBAL STATE & INITIALIZATION
# ============================================================================

class UAP_Manager:
    """Manages UAP generation, storage, and application"""
    
    def __init__(self):
        self.clip_model = None
        self.data_loader = None
        self.generator = None
        self.current_uap = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.workspace_root = str(Path(__file__).parent.parent)
        self.status_log = []
        
    def log(self, message: str):
        """Add status message"""
        self.status_log.append(message)
        print(f"[UAP UI] {message}")
        return "\n".join(self.status_log[-20:])  # Last 20 messages
    
    def initialize_clip_model(self, model_name: str = "ViT-B/32") -> str:
        """Initialize CLIP model"""
        if self.clip_model is not None:
            return self.log("CLIP model already initialized")
        
        if not MODULES_AVAILABLE or not TORCH_AVAILABLE:
            return self.log("CLIP module not available - install torch and clip")
        
        try:
            self.log("Loading CLIP model: {}...".format(model_name))
            self.clip_model = CLIPModelWrapper(model_name, self.device)
            return self.log("CLIP model loaded on {}".format(self.device))
        except Exception as e:
            return self.log("Error loading CLIP model: {}".format(str(e)))
    
    def initialize_data_loader(self, split: str = "val2017") -> str:
        """Initialize COCO data loader"""
        if self.data_loader is not None:
            return self.log("Data loader already initialized ({})".format(split))
        
        if not MODULES_AVAILABLE:
            return self.log("COCO loader module not available")
        
        try:
            coco_dir = os.path.join(self.workspace_root, "data", "MS-COCO", split)
            annotations_file = os.path.join(
                self.workspace_root, "data", "MS-COCO", "annotations", 
                f"captions_{split}.json"
            )
            
            # Check if directory exists
            if not os.path.exists(coco_dir):
                return self.log("COCO directory not found: {}".format(coco_dir))
            
            self.log("Loading COCO dataset from: {}".format(coco_dir))
            self.data_loader = COCODataLoader(
                image_dir=coco_dir if os.path.exists(coco_dir) else None,
                annotations_file=annotations_file if os.path.exists(annotations_file) else None,
                split=split
            )
            return self.log("COCO data loader initialized")
        except Exception as e:
            return self.log("Error initializing data loader: {}".format(str(e)))
    
    def apply_uap_to_image(self, image: np.ndarray, uap_path: str, 
                          alpha: float = 0.7) -> Tuple[np.ndarray, str]:
        """Apply UAP to an image using alpha blending"""
        try:
            if image is None:
                return None, self.log("No image provided")
            
            # Resize image to 224x224 (CLIP standard)
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_pil = image_pil.resize((224, 224))
            image_arr = np.array(image_pil) / 255.0
            
            # Load UAP
            if not os.path.exists(uap_path):
                return None, self.log(f"✗ UAP file not found: {uap_path}")
            
            uap = np.load(uap_path)
            uap = np.transpose(uap.squeeze(), (1, 2, 0))
            
            # Apply alpha blending: new = (1-α)*original + α*(original + uap)
            cloaked = (1 - alpha) * image_arr + alpha * np.clip(image_arr + uap, 0, 1)
            cloaked = np.clip(cloaked, 0, 1)
            
            return cloaked, self.log("UAP applied (alpha={:.2f})".format(alpha))
        except Exception as e:
            return None, self.log("Error applying UAP: {}".format(str(e)))
    
    def compute_fidelity_metrics(self, original: np.ndarray, 
                                cloaked: np.ndarray) -> Tuple[float, float, str]:
        """Compute SSIM and PSNR metrics"""
        try:
            if original is None or cloaked is None:
                return 0, 0, self.log("Missing images for fidelity computation")
            
            if not METRICS_AVAILABLE:
                return 0, 0, self.log("scikit-image not installed - metrics unavailable")
            
            # Resize to match
            if original.shape != cloaked.shape:
                cloaked_pil = Image.fromarray((cloaked * 255).astype(np.uint8))
                cloaked_pil = cloaked_pil.resize((original.shape[1], original.shape[0]))
                cloaked = np.array(cloaked_pil) / 255.0
            
            # Compute metrics
            ssim_val = ssim(original, cloaked, channel_axis=2, data_range=1.0)
            psnr_val = psnr(original, cloaked, data_range=1.0)
            
            status = self.log("SSIM: {:.4f} (target: >0.90), PSNR: {:.2f} dB".format(ssim_val, psnr_val))
            return ssim_val, psnr_val, status
        except Exception as e:
            return 0, 0, self.log("Error computing metrics: {}".format(str(e)))


# Initialize global manager
uap_manager = UAP_Manager()


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def upload_and_display_image(image_file) -> Tuple[np.ndarray, str]:
    """Upload and display image"""
    try:
        if image_file is None:
            return None, "No image uploaded"
        
        img = Image.open(image_file)
        img_array = np.array(img) / 255.0
        
        status = "Image loaded: {} ({})".format(img.size, img.mode)
        return img_array, status
    except Exception as e:
        return None, "Error loading image: {}".format(str(e))


def select_coco_image(sample_index: int) -> Tuple[np.ndarray, str]:
    """Select a random COCO image"""
    try:
        if uap_manager.data_loader is None:
            uap_manager.initialize_data_loader()
        
        if uap_manager.data_loader is None:
            return None, "Data loader not initialized"
        
        # Get random image
        image_paths = uap_manager.data_loader.image_paths
        if not image_paths:
            return None, "No COCO images available"
        
        idx = sample_index % len(image_paths)
        image_path = image_paths[idx]
        
        img = Image.open(image_path)
        img_array = np.array(img) / 255.0
        
        status = "COCO Sample {}: {}".format(idx, os.path.basename(image_path))
        return img_array, status
    except Exception as e:
        return None, "Error selecting COCO image: {}".format(str(e))


def preview_uap_pattern(uap_path: str) -> Tuple[np.ndarray, str]:
    """Preview the UAP pattern"""
    try:
        if not os.path.exists(uap_path):
            return None, "UAP file not found: {}".format(uap_path)
        
        uap = np.load(uap_path)
        uap = np.transpose(uap.squeeze(), (1, 2, 0))
        
        # Visualize with enhancement
        uap_vis = np.clip(uap * 10 + 0.5, 0, 1)
        
        return uap_vis, "UAP pattern loaded (shape: {})".format(uap.shape)
    except Exception as e:
        return None, "Error loading UAP: {}".format(str(e))


def generate_comparison_plot(original: np.ndarray, cloaked: np.ndarray, 
                            uap_vis: np.ndarray = None) -> Tuple[str, str]:
    """Generate side-by-side comparison plot"""
    try:
        if original is None or cloaked is None:
            return None, "Missing images for comparison"
        
        # Create figure
        num_plots = 3 if uap_vis is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        if uap_vis is not None:
            axes[1].imshow(uap_vis)
            axes[1].set_title("The Cloak (UAP Pattern)", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            axes[2].imshow(cloaked)
            axes[2].set_title("Cloaked Image", fontsize=12, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[1].imshow(cloaked)
            axes[1].set_title("Cloaked Image", fontsize=12, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf, "Comparison plot generated"
    except Exception as e:
        return None, "Error generating plot: {}".format(str(e))


def export_cloaked_image(cloaked: np.ndarray, filename: str = "cloaked_image.png") -> Tuple[str, str]:
    """Export cloaked image"""
    try:
        if cloaked is None:
            return None, "No cloaked image to export"
        
        # Create output directory
        output_dir = os.path.join(uap_manager.workspace_root, "data", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        output_path = os.path.join(output_dir, filename)
        img = Image.fromarray((cloaked * 255).astype(np.uint8))
        img.save(output_path)
        
        return str(output_path), uap_manager.log("Image exported to {}".format(output_path))
    except Exception as e:
        return None, uap_manager.log("Error exporting image: {}".format(str(e)))


# Professional CSS Styling
PROFESSIONAL_CSS = """
/* Global Layout */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Header */
.app-header {
    text-align: center;
    padding: 50px 30px;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 0;
}

.app-header h1 {
    margin: 0; 
    font-size: 2.2em;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.app-header p {
    margin: 12px 0 0 0;
    font-size: 1em;
    opacity: 0.95;
    font-weight: 300;
    line-height: 1.6;
}

/* Tabs */
.tabitem {
    background: white;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.tab-nav {
    border-bottom: 2px solid #e5e7eb;
    gap: 8px;
}

.selected {
    border-bottom: 3px solid #3b82f6 !important;
    color: #1f2937 !important;
}

/* Buttons */
.gr-button.primary {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    border: none;
    font-weight: 600;
    padding: 12px 24px !important;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}

.gr-button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.gr-button:not(.primary) {
    border-radius: 8px;
    border: 1.5px solid #d1d5db;
}

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
}

/* Input Fields */
.gr-textbox,
.gr-slider,
.gr-dropdown,
.gr-file {
    border-radius: 8px;
    border: 1.5px solid #d1d5db !important;
    background: white;
}

.gr-textbox:focus,
.gr-slider:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Status Messages */
.status-success {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 16px;
    border-radius: 8px;
}

.status-warning {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 16px;
    border-radius: 8px;
}

.status-error {
    background: #fee2e2;
    border-left: 4px solid #ef4444;
    padding: 16px;
    border-radius: 8px;
}

/* Info Section */
.info-section {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
    border-left: 4px solid #0284c7;
    padding: 24px;
    border-radius: 8px;
    margin: 24px 0;
}

.info-section h3 {
    margin-top: 0;
    color: #0c4a6e;
    font-size: 1.1em;
    font-weight: 600;
}

.info-section p {
    color: #1e293b;
    line-height: 1.6;
    margin: 8px 0;
}

/* Section Title */
.section-title {
    font-size: 1.3em;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 16px;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 12px;
}

/* Table */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
}

table th {
    background: #f3f4f6;
    padding: 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #e5e7eb;
}

table td {
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
}

/* Responsive */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.5em;
    }
    
    .app-header {
        padding: 30px 20px;
    }
}
"""


# ============================================================================
# MAIN GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """Create professional Gradio interface"""
    
    with gr.Blocks(
        title="Image Protection System",
        theme=gr.themes.Soft(),
        css=PROFESSIONAL_CSS
    ) as app:
        
        # Header
        with gr.Column(elem_classes="app-header"):
            gr.Markdown("# Image Protection System")
            gr.Markdown("Advanced Universal Adversarial Perturbation Application for AI-Model Resilience")
        
        # Main Tabs
        with gr.Tabs():
            
            # ========== TAB 1: SYSTEM SETUP ==========
            with gr.Tab("System Configuration"):
                gr.Markdown("## Initialize System Components")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Selection")
                        model_choice = gr.Dropdown(
                            choices=["ViT-B/32", "ViT-L/14"],
                            value="ViT-B/32",
                            label="CLIP Model Variant",
                            info="Select the CLIP vision-language model for feature extraction"
                        )
                        btn_init_clip = gr.Button("Initialize Model", variant="primary", full_width=True)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Configuration")
                        dataset_split = gr.Dropdown(
                            choices=["val2017", "train2017"],
                            value="val2017",
                            label="Dataset Split",
                            info="Choose validation or training dataset"
                        )
                        btn_init_data = gr.Button("Load Dataset", variant="primary", full_width=True)
                
                status_output = gr.Textbox(
                    label="System Status",
                    lines=12,
                    interactive=False,
                    max_lines=20,
                    placeholder="System status messages will appear here..."
                )
                
                # Event handlers
                btn_init_clip.click(
                    fn=uap_manager.initialize_clip_model,
                    inputs=[model_choice],
                    outputs=[status_output]
                )
                
                btn_init_data.click(
                    fn=uap_manager.initialize_data_loader,
                    inputs=[dataset_split],
                    outputs=[status_output]
                )
            
            # ========== TAB 2: IMAGE SELECTION ==========
            with gr.Tab("Image Selection"):
                gr.Markdown("## Select or Upload Image")
                
                with gr.Row():
                    # Custom Upload
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Custom Image")
                        image_upload = gr.File(
                            label="Select Image File",
                            type="filepath",
                            file_types=["image"]
                        )
                        btn_load_custom = gr.Button("Load Custom Image", variant="primary", full_width=True)
                    
                    # COCO Selection
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Browser")
                        sample_index = gr.Slider(
                            minimum=0,
                            maximum=4999,
                            step=1,
                            value=0,
                            label="Dataset Index",
                            info="Select image from dataset (0-4999)"
                        )
                        btn_load_coco = gr.Button("Load Dataset Image", variant="primary", full_width=True)
                
                selected_image = gr.Image(
                    label="Selected Image",
                    type="numpy",
                    interactive=False
                )
                image_status = gr.Textbox(
                    label="Image Information",
                    interactive=False,
                    lines=2
                )
                
                # Event handlers
                btn_load_custom.click(
                    fn=upload_and_display_image,
                    inputs=[image_upload],
                    outputs=[selected_image, image_status]
                )
                
                btn_load_coco.click(
                    fn=select_coco_image,
                    inputs=[sample_index],
                    outputs=[selected_image, image_status]
                )
            
            # ========== TAB 3: APPLY PROTECTION ==========
            with gr.Tab("Apply Protection"):
                gr.Markdown("## Protection Application")
                
                with gr.Row():
                    # Configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        
                        uap_file_path = gr.Textbox(
                            label="UAP File Path",
                            value="data/results/clip_uap_final.npy",
                            placeholder="Path to protection parameters",
                            info="Location of pre-generated protection patterns"
                        )
                        
                        alpha_blending = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.7,
                            label="Protection Strength",
                            info="Controls blending intensity: 0=minimal, 1=maximum"
                        )
                        
                        btn_apply_uap = gr.Button("Apply Protection", variant="primary", full_width=True, size="lg")
                    
                    # Results
                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        cloaked_image = gr.Image(
                            label="Protected Image",
                            type="numpy",
                            interactive=False
                        )
                
                apply_status = gr.Textbox(
                    label="Operation Status",
                    interactive=False,
                    lines=3
                )
                
                # Info
                with gr.Group(elem_classes="info-section"):
                    gr.Markdown("""
                    ### Protection Strength Guidance
                    
                    | Level | Range | Application |
                    |-------|-------|-------------|
                    | **Subtle** | 0.0 - 0.3 | Minimal visual changes |
                    | **Balanced** | 0.4 - 0.7 | General use (recommended) |
                    | **Strong** | 0.8 - 1.0 | Maximum protection |
                    """)
                
                # Event handler
                btn_apply_uap.click(
                    fn=lambda img, path, alpha: (
                        uap_manager.apply_uap_to_image(img, path, alpha)
                        if img is not None else (None, "No image selected. Please upload or load an image first.")
                    ),
                    inputs=[selected_image, uap_file_path, alpha_blending],
                    outputs=[cloaked_image, apply_status]
                )
            
            # ========== TAB 4: QUALITY VALIDATION ==========
            with gr.Tab("Quality Metrics"):
                gr.Markdown("## Fidelity Validation")
                gr.Markdown("Assess visual quality of protected images using standard metrics.")
                
                btn_compute_metrics = gr.Button("Compute Metrics", variant="primary", full_width=True)
                
                with gr.Row():
                    ssim_score = gr.Number(label="SSIM Index", interactive=False, precision=4)
                    psnr_score = gr.Number(label="PSNR (dB)", interactive=False, precision=2)
                
                metrics_status = gr.Textbox(
                    label="Analysis Results",
                    interactive=False,
                    lines=5
                )
                
                gr.Markdown("""
                ### Metric Interpretation
                
                **SSIM (Structural Similarity Index)**
                - Range: 0.0 to 1.0
                - Higher values indicate greater similarity to original
                - Target: > 0.90 for thesis requirements
                
                **PSNR (Peak Signal-to-Noise Ratio)**
                - Range: 0 to infinity (dB)
                - Higher values indicate better quality
                - Typical range: 20-40 dB
                """)
                
                # Event handler
                btn_compute_metrics.click(
                    fn=lambda orig, cloak: (
                        uap_manager.compute_fidelity_metrics(orig, cloak)
                        if orig is not None and cloak is not None
                        else (0, 0, "Missing images. Please complete image selection and protection steps.")
                    ),
                    inputs=[selected_image, cloaked_image],
                    outputs=[ssim_score, psnr_score, metrics_status]
                )
            
            # ========== TAB 5: EXPORT ==========
            with gr.Tab("Export Results"):
                gr.Markdown("## Save Protected Image")
                
                with gr.Row():
                    export_filename = gr.Textbox(
                        label="Output Filename",
                        value="protected_image.png",
                        placeholder="filename.png",
                        info="Name for saved image file"
                    )
                    btn_export = gr.Button("Export Image", variant="primary", full_width=True)
                
                export_path = gr.Textbox(
                    label="Saved Location",
                    interactive=False,
                    lines=1
                )
                export_status = gr.Textbox(
                    label="Export Status",
                    interactive=False,
                    lines=3
                )
                
                # Event handler
                btn_export.click(
                    fn=export_cloaked_image,
                    inputs=[cloaked_image, export_filename],
                    outputs=[export_path, export_status]
                )
            
            # ========== TAB 6: DOCUMENTATION ==========
            with gr.Tab("Documentation"):
                gr.Markdown("""
                ## System Overview
                
                This application implements Universal Adversarial Perturbations (UAP) to protect images from unauthorized AI analysis.
                
                ### How It Works
                
                The system generates imperceptible noise patterns that disrupt the semantic understanding of vision-language models (specifically CLIP).
                These perturbations preserve image quality while degrading AI model performance.
                
                **Mathematical Foundation:**
                
                V(t+1) = Π[V(t) - α·sign(∇ cos(f_img(I + V), f_txt(T)))]
                
                Where: V = perturbation, α = learning rate, Π = projection operator, cos = cosine similarity
                
                ### Key Features
                
                1. **Universal Application**: Single perturbation works across multiple images
                2. **Imperceptible**: Changes are below human perception threshold
                3. **Fidelity Preservation**: Maintains SSIM > 0.90
                4. **Mobile Optimized**: Alpha blending for device compatibility
                
                ### Citation
                
                Capayan et al. (2024). "Mobile-Based Cloaking of Proprietary Images against Unauthorized AI Training"
                West Visayas State University, Computer Science Department
                
                ### References
                
                - Moosavi-Dezfooli et al. (2017): Universal Adversarial Perturbations
                - Radford et al. (2021): CLIP - Learning Transferable Models for Computer Vision
                - Kurakin et al. (2016): Adversarial Examples in the Physical World
    
    return app


# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    app = create_gradio_interface()
    
    print("\n" + "=" * 70)
    print("Image Protection System - Gradio Interface")
    print("=" * 70)
    print("Device: {}".format(uap_manager.device))
    print("Workspace: {}".format(uap_manager.workspace_root))
    print("\nLaunching interface...")
    print("=" * 70 + "\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
