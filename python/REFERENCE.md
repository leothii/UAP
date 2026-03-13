# 🎯 VeilAI - Quick Reference Card

## Installation & Launch

```bash
# Install all dependencies
pip install -r requirements.txt

# Launch with one command
python launch_ui.py --full      # Full-featured
python launch_ui.py --lite      # Lightweight  
python launch_ui.py --demo      # Demo/testing
python launch_ui.py --config    # Show settings
```

**Access UI:** http://localhost:7860

---

## UI Tabs Overview

### 1️⃣ System Setup
```
Initialize CLIP model + COCO dataset
Status: Shows what's ready
```

### 2️⃣ Image Selection  
```
Upload custom image OR pick from COCO dataset
Output: Selected image displayed
```

### 3️⃣ Apply Protection
```
Input: Image + UAP file path + Alpha value
Output: Cloaked image
Alpha: 0.7 (recommended)
```

### 4️⃣ Quality Metrics
```
Compute SSIM & PSNR scores
SSIM target: > 0.90
PSNR: Higher is better
```

### 5️⃣ Visualization
```
Generate before/after/cloak comparison
3-panel figure showing the process
```

### 6️⃣ Export & Batch
```
Save protected image
Batch processing: Coming soon
```

### 7️⃣ Documentation
```
Read algorithm details
View academic references
Team information
```

---

## Key Parameters

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| **Alpha** | 0.7 | 0.0-1.0 | Perturbation opacity |
| **SSIM Target** | 0.90 | 0.85-0.97 | Quality threshold |
| **Batch Size** | 32 | 1-128 | Images per iteration |
| **Learning Rate** | 0.01 | 0.0001-0.1 | Optimization step |
| **Iterations** | 100 | 10-500 | Generation iterations |

---

## Alpha Blending Guide

```
0.0 ───────────────────────── 1.0
↓                              ↓
Original                    Maximum
(no cloak)                   Cloak

0.0-0.3: Invisible
0.3-0.5: Subtle (conservative)
0.5-0.7: Balanced (recommended) ← HERE
0.7-0.9: Strong
0.9-1.0: Maximum visible
```

---

## Command Cheat Sheet

| Task | Command |
|------|---------|
| Start UI | `python launch_ui.py --full` |
| Lightweight version | `python launch_ui.py --lite` |
| Run demo | `python launch_ui.py --demo` |
| Show config | `python launch_ui.py --config` |
| Direct launch | `python gradio_ui.py` |
| Change port | Edit `ui_config.py` line with `server_port` |
| Use CPU | Edit `ui_config.py` set `device = "cpu"` |

---

## File Locations

```
UAP/python/
├── launch_ui.py          ← START HERE
├── gradio_ui.py          ← Full version
├── gradio_ui_lite.py     ← Light version
├── ui_config.py          ← Settings
├── UI_README.md          ← Full docs
├── QUICK_START.md        ← Setup guide
├── REFERENCE.md          ← This file
└── *.py                  ← Supporting modules

UAP/data/
├── results/              ← Exported images
└── MS-COCO/              ← Dataset (optional)
```

---

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| Gradio not found | `pip install gradio --upgrade` |
| CLIP not found | `pip install git+https://github.com/openai/CLIP.git` |
| Port in use | Change port in `ui_config.py` |
| CUDA out of memory | Use CPU or reduce batch size |
| Slow startup | Use `--lite` version |
| No COCO found | Optional - use custom images instead |

---

## Metrics Explained

### SSIM (Structural Similarity Index)
- **Range:** 0.0 to 1.0
- **Higher = Better** (more similar to original)
- **Target:** > 0.90
- **Interpretation:** 
  - 1.0 = Identical
  - 0.90 = Highly similar (thesis req.)
  - 0.70 = Noticeable differences
  - 0.50 = Clear differences

### PSNR (Peak Signal-to-Noise Ratio)
- **Range:** 0 to ∞ dB
- **Higher = Better** (less distortion)
- **Typical:** 20-40 dB
- **Interpretation:**
  - 40+ dB = Excellent quality
  - 30-40 dB = Good quality
  - 20-30 dB = Acceptable
  - <20 dB = Poor quality

---

## Workflow Summary

```
1. pip install -r requirements.txt
   ↓
2. python launch_ui.py --full
   ↓
3. Open http://localhost:7860
   ↓
4. Initialize CLIP + COCO (System Setup tab)
   ↓
5. Upload image (Image Selection tab)
   ↓
6. Apply protection (Apply Protection tab)
   ↓
7. Check quality (Quality Metrics tab)
   ↓
8. View results (Visualization tab)
   ↓
9. Export (Export tab)
```

---

## Performance Tips

✅ **DO:**
- Use GPU if available (much faster)
- Start with `--lite` version for testing
- Use alpha 0.7 (sweet spot)
- Check SSIM > 0.90

❌ **DON'T:**
- Don't regenerate UAP for each image
- Don't set alpha to 0 (defeats purpose)
- Don't use very high iteration counts unless needed
- Don't ignore SSIM threshold

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| RAM | 4 GB | 8 GB |
| GPU | None | NVIDIA 6GB+ |
| Disk | 2 GB | 10 GB |
| Python | 3.8+ | 3.10+ |

---

## Configuration Template

Edit `ui_config.py` to customize:

```python
# Server
GRADIO_CONFIG["server_port"] = 7860
GRADIO_CONFIG["share"] = False

# Model
CLIP_CONFIG["device"] = "cuda"  # or "cpu"
CLIP_CONFIG["model_name"] = "ViT-B/32"

# UAP
UAP_CONFIG["alpha"] = 0.7
UAP_CONFIG["ssim_threshold"] = 0.90

# COCO
COCO_CONFIG["batch_size"] = 32
```

---

## Hot Keys in Browser

- `F5` - Refresh page
- `Ctrl+Shift+K` - Dev console (if needed)
- `Ctrl+C` - Stop server (in terminal)

---

## Support Matrix

| Issue | Solution | Time |
|-------|----------|------|
| Install deps | `pip install -r requirements.txt` | 5-10 min |
| Start UI | `python launch_ui.py --full` | 1-2 min |
| First time wait | CLIP download | ~30s |
| Load COCO | Dataset indexing | ~30s |
| Apply UAP | Instant | <1s |

---

## Version Information

- **UI Version:** 1.0
- **Gradio:** ≥4.0.0
- **Python:** ≥3.8
- **Status:** ✅ Stable

---

**Need help?** See UI_README.md or QUICK_START.md
