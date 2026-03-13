# Gradio UI for VeilAI

Complete web interface for VeilAI - the image protection system that prevents unauthorized AI training, built with Gradio.

## 🚀 Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install just Gradio for lightweight version
pip install gradio
```

### Launch UI

```bash
# Full-featured UI (requires all dependencies)
python launch_ui.py --full

# Lightweight UI (minimal dependencies, pre-generated UAPs)
python launch_ui.py --lite

# Demo UI (no dependencies, testing only)
python launch_ui.py --demo

# Show configuration
python launch_ui.py --config
```

The UI will be available at: `http://localhost:7860`

---

## 📋 Features Overview

### Three UI Versions

#### 1. **Full-Featured UI** (`gradio_ui.py`)
**Use when:** You have all dependencies and want complete functionality

**Features:**
- Complete CLIP model initialization
- COCO dataset integration
- UAP generation from scratch
- Real-time visualization
- Fidelity validation (SSIM, PSNR)
- Batch processing (planned)
- Export to mobile formats

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
pillow>=9.0.0
matplotlib>=3.5.0
scikit-image>=0.19.0
gradio>=4.0.0
clip @ git+https://github.com/openai/CLIP.git
```

#### 2. **Lightweight UI** (`gradio_ui_lite.py`)
**Use when:** You have pre-generated UAPs and want a simple interface

**Features:**
- Quick image upload
- Apply pre-computed UAP
- Adjust alpha blending
- Download protected image
- Minimal dependencies

**Requirements:**
```
gradio>=4.0.0
numpy>=1.20.0
pillow>=9.0.0
```

#### 3. **Demo UI** (via `launch_ui.py --demo`)
**Use when:** Testing without any actual dependencies

**Features:**
- Test UI/UX
- No actual processing
- No dependencies needed

---

## 🎯 Detailed Workflow

### Tab 1: System Setup
Initialize core components before running:

```
✓ CLIP Model (ViT-B/32 or ViT-L/14)
✓ COCO Dataset Loader (val2017 or train2017)
✓ GPU/CPU Detection
```

**Status Examples:**
- ✅ CLIP model loaded on cuda
- ✅ COCO data loader initialized (5000 images found)

### Tab 2: Image Selection
Choose your input image:

**Option A: Custom Upload**
```
1. Click "Upload Image"
2. Select local file (JPG, PNG, etc.)
3. Click "Load Custom Image"
```

**Option B: COCO Dataset**
```
1. Adjust sample index slider (0-4999)
2. Click "Load COCO Sample"
3. Random image will be selected
```

### Tab 3: Apply Protection
Generate and apply UAP:

**Parameters:**
- **UAP File Path**: Location of .npy file (default: `data/results/clip_uap_final.npy`)
- **Alpha Blending**: 0.0-1.0 (default: 0.7)

**Alpha Guidelines:**
```
0.0-0.2  : Invisible (minimal protection)
0.3-0.5  : Subtle (recommended for conservative)
0.6-0.8  : Balanced (recommended for general)
0.9-1.0  : Aggressive (maximum visibility)
```

**Output:**
- UAP Pattern visualization
- Cloaked image result
- Processing status

### Tab 4: Quality Metrics
Validate the protected image:

**Metrics Computed:**
- **SSIM** (Structural Similarity Index)
  - Range: 0.0 to 1.0
  - Target: > 0.90 (thesis requirement)
  - Interpretation: Higher = more similar to original

- **PSNR** (Peak Signal-to-Noise Ratio)
  - Range: 0-∞ dB
  - Typical: 20-40 dB
  - Interpretation: Higher = less distortion

### Tab 5: Visualization
Generate side-by-side comparison plots:

**Shows:**
1. Original image (unprotected)
2. UAP pattern (the "cloak")
3. Final cloaked image

### Tab 6: Export & Batch
Save results and process multiple images:

**Single Export:**
```
1. Enter filename (default: cloaked_image.png)
2. Click "Export Image"
3. Gets saved to data/results/
```

**Batch Processing:** (Coming Soon)
- Upload multiple images
- Apply same UAP to all
- Batch download results

### Tab 7: Documentation
Reference materials and project info:

- Technical explanation
- Algorithm details  
- Paper references
- Team information

---

## 📁 Project Structure

```
UAP/
├── python/
│   ├── gradio_ui.py              # Main full-featured interface
│   ├── gradio_ui_lite.py         # Lightweight interface
│   ├── launch_ui.py              # Launch script (recommended entry point)
│   ├── ui_config.py              # Centralized configuration
│   ├── clip_integration.py       # CLIP model wrapper
│   ├── clip_uap_generator.py     # UAP generation engine
│   ├── coco_loader.py            # COCO dataset loader
│   ├── fidelity_validator.py     # Quality validation
│   ├── visualize_cloak.py        # Visualization utilities
│   ├── requirements.txt          # Python dependencies
│   └── UI_README.md              # This file
├── data/
│   ├── MS-COCO/
│   │   ├── val2017/              # Validation images
│   │   └── annotations/          # Captions
│   └── results/
│       └── clip_uap_final.npy    # Pre-generated UAP
└── README.md                     # Main project documentation
```

---

## ⚙️ Configuration

### Using `ui_config.py`

All settings are centralized in `ui_config.py`:

```python
# Server settings
HOST = "0.0.0.0"
PORT = 7860
SHARE = False

# Model settings
CLIP_MODEL = "ViT-B/32"
DEVICE = "auto"  # cuda or cpu

# UAP parameters
ALPHA = 0.7
LEARNING_RATE = 0.01
NUM_ITERATIONS = 100
SSIM_THRESHOLD = 0.90

# Presets
PRESETS = {
    "conservative": {...},
    "balanced": {...},
    "aggressive": {...}
}
```

### Print Current Config

```bash
python launch_ui.py --config
```

Output example:
```
[PATHS]
  Project Root: /path/to/UAP
  Data Dir: /path/to/UAP/data

[CLIP MODEL]
  model_name: ViT-B/32
  device: cuda

[UAP GENERATION]
  alpha: 0.7
  learning_rate: 0.01
  ssim_threshold: 0.90
```

---

## 🔧 Advanced Usage

### Custom Configuration

Edit `ui_config.py` to customize:

```python
# Change server port
GRADIO_CONFIG["server_port"] = 8000

# Change alpha behavior
UAP_CONFIG["alpha"] = 0.5
UAP_CONFIG["alpha_step"] = 0.1

# Change dataset
COCO_CONFIG["splits"] = ["train2017"]
```

### Disable Gradio Analytics

```python
# In launch_ui.py before launching
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
```

### Public Access

```python
# Share externally (temporary link)
GRADIO_CONFIG["share"] = True

# Run on custom IP
GRADIO_CONFIG["server_name"] = "192.168.1.100"
```

---

## 📊 Performance & Optimization

### Memory Requirements

| Component | RAM | GPU |
|-----------|-----|-----|
| CLIP Model | 1GB | 2GB |
| COCO Mini-batch | 2GB | 4GB |
| Total | 3-5GB | 6-8GB |

### Optimization Tips

1. **Reduce batch size** if running on limited RAM
2. **Use `--lite` version** with pre-generated UAPs
3. **Disable gradio analytics** for faster startup
4. **Use CPU if GPU not available** (slower but works)

### Profiling

```python
import cProfile
import gradio_ui

cProfile.run('gradio_ui.create_gradio_interface()')
```

---

## 🐛 Troubleshooting

### Issue: "CLIP module not found"

```bash
# Solution: Install CLIP separately
pip install git+https://github.com/openai/CLIP.git
```

### Issue: "COCO directory not found"

```bash
# Ensure dataset structure:
data/
└── MS-COCO/
    ├── val2017/          # Download images here
    └── annotations/      # Download JSON here
```

[See main README.md for COCO download instructions]

### Issue: "Gradio port already in use"

```bash
# Use different port in launch_ui.py:
GRADIO_CONFIG["server_port"] = 7861
```

### Issue: "Out of memory during generation"

```python
# Reduce batch size in ui_config.py:
COCO_CONFIG["batch_size"] = 16
```

### Issue: "CUDA out of memory"

```bash
# Use CPU instead:
python gradio_ui.py --device cpu
```

---

## 📚 API Reference

### Main Functions

#### `create_gradio_interface()`
Creates and returns the full Gradio Blocks interface.

```python
from gradio_ui import create_gradio_interface
app = create_gradio_interface()
app.launch()
```

#### `UAP_Manager`
Central manager class for UAP operations.

```python
from gradio_ui import uap_manager

# Initialize components
uap_manager.initialize_clip_model("ViT-B/32")
uap_manager.initialize_data_loader("val2017")

# Apply UAP
cloaked, status = uap_manager.apply_uap_to_image(
    image=image_array,
    uap_path="data/results/clip_uap_final.npy",
    alpha=0.7
)

# Compute metrics
ssim, psnr, status = uap_manager.compute_fidelity_metrics(original, cloaked)
```

#### `upload_and_display_image(image_file)`
Handle image upload from file.

```python
image_array, status = upload_and_display_image("/path/to/image.jpg")
```

#### `select_coco_image(sample_index)`
Get image from COCO dataset.

```python
image_array, status = select_coco_image(42)
```

---

## 🎓 Educational Notes

This UI demonstrates:

### 1. **Adversarial ML Concepts**
- Universal Adversarial Perturbations (UAP)
- Semantic attacks on vision-language models
- Gradient-based optimization

### 2. **Practical Deep Learning**
- Model loading and inference
- Batch processing
- GPU/CPU compatibility

### 3. **Web UI Development**
- Gradio for rapid prototyping
- State management
- File handling
- Real-time visualization

### 4. **Research Workflow**
- Data pipeline design
- Metric computation
- Result visualization
- Reproducibility

---

## 📝 Citation

If you use this UI in your research, please cite:

```bibtex
@article{capayan2024uap,
  title={Mobile-Based Cloaking of Proprietary Images against Unauthorized AI Training},
  author={Capayan, Quinjie Benedict and Chua, Ralph Martin and Diana, Gabriel and Libuna, Donjie C.},
  school={West Visayas State University},
  year={2024}
}
```

---

## 📞 Support

### Common Questions

**Q: Can I use this on mobile?**
A: The UI runs on desktop. But the generated UAPs are designed for mobile app integration.

**Q: How long does UAP generation take?**
A: ~2-5 minutes on GPU, ~30-60 minutes on CPU (depends on iterations).

**Q: Do I need to regenerate UAP for each image?**
A: No! The UAP is universal - same perturbation works for ALL images.

**Q: What's the practical use case?**
A: Protect your artwork/content from being scraped for AI training.

### Getting Help

1. Check [Troubleshooting](#-troubleshooting) section
2. Review [Configuration](#-configuration) section
3. Run with `--debug` flag for verbose output
4. Check main [README.md](../README.md)

---

## 🔄 Updates & Roadmap

### Current Version: 1.0
- ✅ Image upload and COCO selection
- ✅ UAP application with alpha blending
- ✅ Fidelity validation
- ✅ Visualization and export
- ✅ Lightweight and full versions

### Planned Features
- ⏳ Batch processing UI
- ⏳ Real-time UAP generation
- ⏳ Mobile app integration preview
- ⏳ Comparison with other defense methods
- ⏳ Dark mode
- ⏳ Multi-language support

---

**Last Updated:** 2024  
**Version:** 1.0  
**Status:** Stable ✅
