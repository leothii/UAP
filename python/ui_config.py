"""
Configuration module for Gradio UI
Centralized settings for the entire application
"""

from pathlib import Path
import os
from typing import Dict, Any

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_DIR = PROJECT_ROOT / "python"
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "MS-COCO"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


# ============================================================================
# GRADIO SERVER SETTINGS
# ============================================================================

GRADIO_CONFIG = {
    # Server
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,  # Set to True for public sharing
    "show_error": True,
    "debug": False,
    
    # Interface
    "theme": "soft",
    "title": "🔒 VeilAI - Image Protection System",
    "description": "Protect your images from unauthorized AI training",
    "show_api": True,
    
    # Browser
    "inbrowser": False,  # Auto-open browser
}


# ============================================================================
# MODEL SETTINGS
# ============================================================================

CLIP_CONFIG = {
    "model_name": "ViT-B/32",  # Also supports "ViT-L/14"
    "device": None,  # None = auto-detect, "cpu" = force CPU, "cuda" = force GPU
    "dtype": "float32",
}


# ============================================================================
# DATASET SETTINGS
# ============================================================================

COCO_CONFIG = {
    "splits": ["val2017", "train2017"],
    "val2017_size": 5000,  # Number of validation images
    "train2017_size": 118287,  # Number of training images
    "image_size": 224,  # Standard CLIP input size
    "batch_size": 32,  # For mini-batching during UAP generation
}


# ============================================================================
# UAP GENERATION SETTINGS
# ============================================================================

UAP_CONFIG = {
    # Alpha blending
    "alpha": 0.7,  # Default blending factor for mobile
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_step": 0.05,
    
    # Optimization
    "learning_rate": 0.01,  # Step size for gradient descent
    "num_iterations": 100,
    "epsilon": 8.0 / 255,  # Max perturbation magnitude
    
    # Fidelity constraints
    "ssim_threshold": 0.90,  # Must maintain SSIM > 0.90
    "psnr_threshold": 30.0,  # Minimum acceptable PSNR (dB)
    
    # File paths
    "uap_output_path": str(RESULTS_DIR / "clip_uap_final.npy"),
    "checkpoint_dir": str(RESULTS_DIR / "checkpoints"),
}


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VIZ_CONFIG = {
    "figure_size": (15, 5),
    "dpi": 100,
    "save_format": "png",
    "colormap": "viridis",
}


# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    "log_file": str(PROJECT_ROOT / "gradio_ui.log"),
}


# ============================================================================
# APPLICATION PRESETS
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "description": "Subtle protection, minimal visual change",
        "alpha": 0.3,
        "num_iterations": 50,
        "ssim_threshold": 0.97,
    },
    "balanced": {
        "description": "Recommended for general use",
        "alpha": 0.7,
        "num_iterations": 100,
        "ssim_threshold": 0.90,
    },
    "aggressive": {
        "description": "Maximum protection, more visible changes",
        "alpha": 1.0,
        "num_iterations": 150,
        "ssim_threshold": 0.85,
    },
}


# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALIDATION_RANGES = {
    "alpha": (0.0, 1.0),
    "learning_rate": (0.0001, 0.1),
    "num_iterations": (10, 500),
    "epsilon": (0.01, 0.1),
    "batch_size": (1, 128),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(key: str, default=None) -> Any:
    """Get config value by dot notation (e.g., 'CLIP_CONFIG.model_name')"""
    
    configs = {
        "gradio": GRADIO_CONFIG,
        "clip": CLIP_CONFIG,
        "coco": COCO_CONFIG,
        "uap": UAP_CONFIG,
        "viz": VIZ_CONFIG,
        "logging": LOGGING_CONFIG,
    }
    
    if "." in key:
        config_type, config_key = key.split(".", 1)
        return configs.get(config_type, {}).get(config_key, default)
    
    return default


def print_config():
    """Print all configuration values"""
    
    print("\n" + "="*70)
    print("GRADIO UI CONFIGURATION")
    print("="*70)
    
    print("\n[PATHS]")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python Dir: {PYTHON_DIR}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  COCO Dir: {COCO_DIR}")
    print(f"  Results Dir: {RESULTS_DIR}")
    
    print("\n[GRADIO SERVER]")
    for k, v in GRADIO_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n[CLIP MODEL]")
    for k, v in CLIP_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n[COCO DATASET]")
    for k, v in COCO_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n[UAP GENERATION]")
    for k, v in UAP_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_config()
