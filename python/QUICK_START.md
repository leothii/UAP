# 🚀 Quick Start Guide - Running VeilAI

Get the VeilAI image protection system up and running in minutes!

---

## Step 1: Install Dependencies

```bash
# Navigate to python directory
cd python

# Install all requirements
pip install -r requirements.txt
```

**Installation time:** ~5-10 minutes (depending on internet speed)

**Check installation:**
```bash
python -c "import gradio; import torch; print('✓ All dependencies installed')"
```

---

## Step 2: Launch the UI

### Recommended: Use Launch Script

```bash
# Full-featured UI (requires all dependencies)
python launch_ui.py --full

# Light-weight UI (faster startup)
python launch_ui.py --lite

# Demo (no dependencies)
python launch_ui.py --demo

# Show configuration
python launch_ui.py --config
```

### Or Direct Python

```bash
# Full version
python gradio_ui.py

# Lightweight
python gradio_ui_lite.py
```

---

## Step 3: Access the UI

Open your browser and go to:
```
http://localhost:7860
```

You should see the Gradio interface with multiple tabs!

---

## Common First-Time Issues & Fixes

### ❌ "ModuleNotFoundError: No module named 'gradio'"

```bash
# Install Gradio
pip install gradio --upgrade
```

### ❌ "No module named 'clip'"

```bash
# Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git
```

### ❌ "COCO directory not found"

The UI works fine without COCO! It will:
- Skip COCO loading
- Let you upload custom images
- Still apply pre-generated UAP

### ❌ Port 7860 already in use

Edit `ui_config.py`:
```python
GRADIO_CONFIG["server_port"] = 7861  # Or any free port
```

### ❌ "CUDA out of memory"

```bash
# Use CPU instead
# Edit ui_config.py:
CLIP_CONFIG["device"] = "cpu"
```

---

## Workflow Example

### Using Full UI:

1. **System Setup Tab**
   - Click "Initialize CLIP Model" ✓
   - Click "Initialize COCO Loader" ✓

2. **Image Selection Tab**
   - Click "Load COCO Sample" OR
   - Upload your own image

3. **Apply Protection Tab**
   - Adjust alpha slider (try 0.7)
   - Click "Apply Protection" ✓

4. **Quality Metrics Tab**
   - Click "Compute Metrics" ✓
   - View SSIM and PSNR scores

5. **Export Tab**
   - Click "Export Image" ✓
   - Find result in `data/results/`

---

## File Locations

Important paths to know:

```
UAP/
├── python/
│   ├── launch_ui.py              ← Run this!
│   ├── gradio_ui.py              ← Full interface
│   ├── gradio_ui_lite.py         ← Light interface  
│   ├── ui_config.py              ← Settings
│   ├── UI_README.md              ← Full documentation
│   └── requirements.txt           ← Dependencies
│
├── data/
│   ├── results/
│   │   └── clip_uap_final.npy    ← Generated UAP
│   └── MS-COCO/                  ← Dataset (optional)
│
└── README.md                     ← Main project docs
```

---

## Understanding the Tabs

| Tab | Purpose | What It Does |
|-----|---------|------------|
| 🔧 System Setup | Initialize | Load CLIP model and dataset |
| 🖼️ Image Selection | Input | Choose or upload image |
| ⚡ Apply Protection | Process | Generate/apply UAP |
| 📊 Quality Metrics | Validate | Check SSIM and PSNR |
| 🎨 Visualization | View | See before/after comparison |
| 💾 Export | Save | Download protected image |
| 📚 Documentation | Learn | Read about the technology |

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Launch UI: `python launch_ui.py --full`
3. ✅ Open browser: `http://localhost:7860`
4. ✅ Initialize CLIP and COCO
5. ✅ Upload an image
6. ✅ Apply protection
7. ✅ Export result

---

## Configuration & Customization

Want to tweak settings? Edit `ui_config.py`:

```python
# Change server port
GRADIO_CONFIG["server_port"] = 8000

# Change alpha strength
UAP_CONFIG["alpha"] = 0.5

# Change CLIP model
CLIP_CONFIG["model_name"] = "ViT-L/14"

# Use CPU instead of GPU
CLIP_CONFIG["device"] = "cpu"
```

See **UI_README.md** for full config options.

---

## Keyboard Shortcuts

Once in the browser:

- **Ctrl+C**: Stop server in terminal
- **F5 or Ctrl+R**: Refresh UI
- **Ctrl+Shift+K**: Open developer console (if needed)

---

## Need More Help?

📖 Read the full documentation: [UI_README.md](UI_README.md)

🐛 Check troubleshooting section in [UI_README.md](UI_README.md#-troubleshooting)

📚 Learn about the project: [../README.md](../README.md)

---

## Tips & Tricks

### 💡 Speed Up with Lite Version
```bash
python launch_ui.py --lite
```
→ Faster startup, works with pre-generated UAPs

### 💡 Run Without Breaking
```bash
# Terminal 1:
python launch_ui.py --full

# Terminal 2 (separate):
# Do other work while UI runs
```

### 💡 Share UI Publicly
In `ui_config.py`:
```python
GRADIO_CONFIG["share"] = True  # Creates temp public link
```

### 💡 Test Without CPU/GPU Concerns
```bash
python launch_ui.py --demo
```
→ UI works offline, no actual processing

---

## Performance Expectations

| Operation | Time | Hardware |
|-----------|------|----------|
| Startup | 5-10s | Any |
| Load CLIP model | 30-60s | GPU faster |
| Initialize COCO | 10-30s | Depends on SSD |
| Apply UAP | <1s | Instant |
| Compute metrics | 1-2s | Fast |
| Generate plot | 1-2s | Fast |

**Total first-time setup:** ~2-3 minutes with GPU

---

## You're All Set! 🎉

Your Gradio UI is now ready to:
- ✅ Upload images
- ✅ Generate/apply UAP cloak
- ✅ Validate quality
- ✅ Export results
- ✅ Learn about adversarial ML

**Now go protect those images!**

---

For detailed information, see [UI_README.md](UI_README.md)
