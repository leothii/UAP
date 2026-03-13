# 🚀 How to Run VeilAI

Complete step-by-step guide to launch and use the VeilAI Image Protection System

---

## 📋 Table of Contents

1. [Quick Start (2 Minutes)](#quick-start)
2. [First Time Setup](#first-time-setup)
3. [Running VeilAI](#running-veilai)
4. [Accessing the Web Interface](#accessing-the-web-interface)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### For the Impatient ⚡

Open PowerShell/Terminal and run:

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --lite
```

Then open your browser to: **http://localhost:7860**

Done! Start uploading images and applying VeilAI protection.

---

## First Time Setup

### Step 1: Install Dependencies

Open PowerShell or Command Prompt and navigate to the project:

```bash
cd c:\Users\Acer\Documents\Programs\UAP
```

Install all required packages:

```bash
pip install -r python/requirements.txt
```

**What this does:**
- Installs Gradio (web framework)
- Installs PyTorch (deep learning)
- Installs CLIP (vision-language model)
- Installs image processing libraries

**⏱️ Takes 3-5 minutes** (depends on internet speed)

---

## Running VeilAI

### Option 1: Lightweight Version (Recommended) 🎯

**Best for:** Quick testing, minimal setup

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --lite
```

**Features:**
- ✅ Fast startup (no model loading)
- ✅ Minimal dependencies
- ✅ Perfect for trying out image protection
- ❌ No CLIP model initialization

---

### Option 2: Full-Featured Version (Production) 🔬

**Best for:** Complete system with all features

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --full
```

**Features:**
- ✅ Full CLIP model integration
- ✅ COCO dataset loading
- ✅ Advanced metrics (SSIM, PSNR)
- ❌ Requires all dependencies
- ❌ Slower startup

**⏱️ First run takes 30-60 seconds** (downloading CLIP model ~350MB)

---

### Option 3: Demo Version (No Dependencies) 🎪

**Best for:** Testing without installing anything

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --demo
```

**Features:**
- ✅ Works with just Gradio
- ✅ Instant startup
- ❌ Simulated UAP (not real protection)

---

## Accessing the Web Interface

### After Running VeilAI

You'll see output like:
```
📱 Launching VeilAI - Lightweight Version...
This version has minimal dependencies

Running on local URL: http://127.0.0.1:7860
```

### Open in Browser

Click the link or manually open:
- **Local:** http://localhost:7860
- **Alternative:** http://127.0.0.1:7860

### What You'll See

```
┌─────────────────────────────────────┐
│  🔒 VeilAI - Image Protection System │
│  Advanced Universal Adversarial...   │
└─────────────────────────────────────┘

┌─ Input Configuration ─────────────┐
│ [Image Upload Area]               │
│ Protection Strength: [====●==] 70% │
└───────────────────────────────────┘

┌─ Protected Result ────────────────┐
│ [Cloaked Image Preview]           │
│ Status: Ready                      │
└───────────────────────────────────┘
```

---

## Using VeilAI

### Basic Workflow

1. **Upload Image**
   - Click the image upload area
   - Select a JPG, PNG, or GIF file
   - Wait for image to load

2. **Adjust Protection**
   - Use the "Protection Strength" slider
   - Range: 0% (no protection) to 100% (max protection)
   - Recommended: 50-80%

3. **View Result**
   - Cloaked image appears on the right
   - Status shows processing progress
   - Download protected image

4. **Download**
   - Right-click the result image
   - Select "Save image as..."
   - Save to your desired location

---

## Troubleshooting

### Problem: "Port 7860 is already in use"

**Solution:** Another VeilAI instance is running

```bash
# Option 1: Close the other terminal/instance
# Option 2: Use a different port
python launch_ui.py --lite --port 7861
```

---

### Problem: "ModuleNotFoundError: No module named 'gradio'"

**Solution:** Dependencies not installed

```bash
pip install -r requirements.txt
```

---

### Problem: "CUDA out of memory" or very slow

**Solution:** Using full version on weak GPU

```bash
# Switch to lightweight version
python launch_ui.py --lite

# Or run full version on CPU (slower but works)
python launch_ui.py --full --cpu
```

---

### Problem: Interface won't load in browser

**Solution 1:** Check the port is correct
- Default: http://localhost:7860
- Check terminal output for actual port

**Solution 2:** Try a different browser
- Chrome, Firefox, Edge all work
- Try incognito/private mode

**Solution 3:** Firewall blocking
- Windows: Allow Python through firewall
- Or disable firewall temporarily (testing only)

---

## Advanced Usage

### View Current Configuration

```bash
python launch_ui.py --config
```

Shows all settings:
- CLIP model variant (ViT-B/32, ViT-L/14)
- Dataset paths
- Alpha blending parameters
- SSIM threshold (quality check)

---

### Stop VeilAI

Press `Ctrl+C` in the terminal where it's running.

```
Ctrl + C
```

This will:
- Shut down the Gradio server
- Free up port 7860
- Close all connections

---

### Run from Anywhere

You don't need to navigate to the python folder if you set up your PATH:

```bash
# From any location:
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --lite
```

Or create a shortcut (Windows):
```batch
@echo off
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --lite
pause
```

Save as `run_veilai.bat` and double-click to launch!

---

## Quick Reference

| Command | Purpose | Speed | Dependencies |
|---------|---------|-------|--------------|
| `python launch_ui.py --lite` | Quick testing | ⚡ Fast | ✓ Minimal |
| `python launch_ui.py --full` | All features | 🐌 Slow | ✓ All required |
| `python launch_ui.py --demo` | No setup | ⚡ Instant | ✓ Gradio only |
| `python launch_ui.py --config` | Show settings | ⚡ Instant | ✓ None |

---

## Environment

**System:** Windows 10/11  
**Python Version:** 3.8+  
**Installed Location:** `c:\Users\Acer\Documents\Programs\UAP`  
**Virtual Environment:** `.venv` folder (auto-created)

---

## Next Steps

- 📖 Read [../python/UI_README.md](../python/UI_README.md) for detailed features
- 🎯 Check [../python/REFERENCE.md](../python/REFERENCE.md) for quick lookup
- 🎨 See [../python/VISUAL_GUIDE.md](../python/VISUAL_GUIDE.md) for architecture
- ⚡ Try [../python/QUICK_START.md](../python/QUICK_START.md) for quick setup

---

## Support

**Issue?** Check the [Troubleshooting](#troubleshooting) section above.

**Still stuck?** Review the error message in the terminal - it usually tells you exactly what's wrong!

---

**Last Updated:** March 13, 2026  
**VeilAI Version:** 1.0 (Professional Lite)
