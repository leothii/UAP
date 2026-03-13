"""
Lightweight Gradio UI for VeilAI - Professional Version
Optimized interface for image protection with advanced styling
"""

import gradio as gr
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple


# Professional CSS Styling
CUSTOM_CSS = """
/* Global Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 0;
}

.header-container h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.header-container p {
    margin: 10px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
    font-weight: 300;
}

/* Card Styling */
.control-panel {
    background: white;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    border: 1px solid #e5e7eb;
}

.result-panel {
    background: white;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    border: 1px solid #e5e7eb;
}

/* Section Headers */
.section-header {
    font-size: 1.3em;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 20px;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 12px;
}

/* Button Styling */
.gr-button.primary {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    border: none;
    font-size: 1em;
    font-weight: 600;
    padding: 12px 24px !important;
    border-radius: 8px;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}

.gr-button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

/* Input Fields */
.gr-textbox,
.gr-slider {
    border-radius: 8px;
    border: 1.5px solid #d1d5db;
    background: white;
}

.gr-textbox:focus,
.gr-slider:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Image Container */
.image-container {
    border: 2px dashed #d1d5db;
    border-radius: 12px;
    padding: 20px;
    background: #fafbfc;
    transition: all 0.3s ease;
}

.image-container:hover {
    border-color: #3b82f6;
    background: #f0f4ff;
}

/* Status Box */
.status-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 16px;
    border-radius: 8px;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
}

.status-box.warning {
    background: #fef3c7;
    border-left-color: #f59e0b;
}

.status-box.error {
    background: #fee2e2;
    border-left-color: #ef4444;
}

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
    border-left: 4px solid #0284c7;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
}

.info-box h3 {
    margin-top: 0;
    color: #0c4a6e;
    font-size: 1.1em;
}

.info-box p {
    margin: 8px 0;
    color: #1e293b;
    line-height: 1.6;
}

/* Slider Label */
.slider-label {
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
}

/* Divider */
hr {
    margin: 30px 0;
    border: none;
    border-top: 1px solid #e5e7eb;
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px 20px;
    color: #6b7280;
    font-size: 0.95em;
    border-top: 1px solid #e5e7eb;
    margin-top: 40px;
}

/* Responsive */
@media (max-width: 768px) {
    .header-container h1 {
        font-size: 1.8em;
    }
    
    .control-panel,
    .result-panel {
        padding: 20px;
    }
}
"""


class ProfessionalUAPUI:
    """Professional UI manager for UAP operations"""
    
    def __init__(self):
        self.workspace_root = str(Path(__file__).parent.parent)
        self.uap_loaded = False
    
    def process_image(self, image_array: np.ndarray, alpha: float = 0.7) -> Tuple[np.ndarray, str]:
        """Apply protection with professional status reporting"""
        
        try:
            if image_array is None:
                return None, "No image provided. Please upload an image to proceed."
            
            # Normalize and prepare
            if image_array.max() > 1:
                image_array = image_array / 255.0
            
            # Resize to 224x224 for CLIP compatibility
            img_pil = Image.fromarray((image_array * 255).astype(np.uint8))
            img_pil = img_pil.resize((224, 224))
            img_normalized = np.array(img_pil) / 255.0
            
            # Load or generate UAP
            uap_path = os.path.join(self.workspace_root, "data", "results", "clip_uap_final.npy")
            
            if not os.path.exists(uap_path):
                uap = np.random.normal(0, 0.05, (3, 224, 224))
                status = "Using generated protection pattern\nAlpha: {:.2f}".format(alpha)
            else:
                uap = np.load(uap_path)
                status = "Protection applied successfully\nAlpha: {:.2f}".format(alpha)
                self.uap_loaded = True
            
            # Shape handling
            if uap.ndim == 4:
                uap = uap.squeeze()
            if uap.shape[0] == 3:
                uap = np.transpose(uap, (1, 2, 0))
            
            # Apply alpha blending
            protected = (1 - alpha) * img_normalized + alpha * np.clip(img_normalized + uap, 0, 1)
            protected = np.clip(protected, 0, 1)
            
            return protected, status
        
        except Exception as e:
            return None, "Error: {}".format(str(e))


def create_professional_interface():
    """Create professional Gradio interface"""
    
    ui = ProfessionalUAPUI()
    
    with gr.Blocks(
        title="Image Protection System",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as app:
        
        # Header
        with gr.Column(elem_classes="header-container"):
            gr.Markdown("# Image Protection System")
            gr.Markdown("Advanced Universal Adversarial Perturbation Application")
        
        # Main Content
        with gr.Row():
            # Left Panel - Controls
            with gr.Column(scale=1, min_width=400):
                with gr.Group(elem_classes="control-panel"):
                    gr.Markdown("## Input Configuration")
                    
                    image_input = gr.Image(
                        label="Select Image",
                        type="numpy",
                        container=True,
                        show_label=True
                    )
                    
                    gr.Markdown("### Protection Settings")
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.7,
                        label="Protection Strength",
                        info="Controls the intensity of applied protection (0 = minimal, 1 = maximum)"
                    )
                    
                    apply_button = gr.Button(
                        "Apply Protection",
                        variant="primary",
                        size="lg"
                    )
            
            # Right Panel - Results
            with gr.Column(scale=1, min_width=400):
                with gr.Group(elem_classes="result-panel"):
                    gr.Markdown("## Protected Result")
                    
                    image_output = gr.Image(
                        label="Protected Image",
                        type="numpy",
                        container=True,
                        show_label=True
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                        container=True,
                        show_label=True
                    )
        
        # Information Section
        with gr.Group(elem_classes="info-box"):
            gr.Markdown("""
            ### Protection Strength Reference
            
            | Level | Range | Use Case |
            |-------|-------|----------|
            | **Subtle** | 0.0 - 0.3 | Minimal visual change |
            | **Balanced** | 0.4 - 0.7 | Recommended for general use |
            | **Strong** | 0.8 - 1.0 | Maximum protection |
            
            **How it works:** The system applies mathematically computed perturbations to disrupt AI model interpretation while preserving image quality.
            """)
        
        # Event Handler
        apply_button.click(
            fn=ui.process_image,
            inputs=[image_input, alpha_slider],
            outputs=[image_output, status_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **System Information**
        
        Input size: 224×224 pixels | Processing: Real-time | Format: PNG/JPG supported
        """)
    
    return app


if __name__ == "__main__":
    app = create_professional_interface()
    print("\nImage Protection System - Professional Interface")
    print("=" * 50)
    print("Server starting on: http://127.0.0.1:7860")
    print("=" * 50 + "\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
