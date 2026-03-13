# 📚 VeilAI Instructions & Setup Guide

Welcome to the VeilAI Instructions folder! Everything you need to run, use, and troubleshoot VeilAI.

---

## 🚀 Start Here

### First Time Running VeilAI?

**Option 1: Easy Way (Windows Users)**
1. Open this folder: `c:\Users\Acer\Documents\Programs\UAP\instructions`
2. **Double-click:** `Start_VeilAI_Lite.bat`
3. Your browser opens automatically
4. Done! ✅

**Option 2: Command Line (All Platforms)**
1. Open PowerShell/Terminal
2. Run:
   ```bash
   cd c:\Users\Acer\Documents\Programs\UAP\python
   python launch_ui.py --lite
   ```
3. Open browser to: http://localhost:7860
4. Done! ✅

---

## 📖 Documentation Files

### [RUN_VEILAI.md](RUN_VEILAI.md) 📋
**Complete step-by-step guide** with everything explained.
- ✅ First time setup
- ✅ Different ways to run VeilAI
- ✅ Detailed troubleshooting
- ✅ Advanced usage tips
- 📖 **Read this for:** Complete understanding, setup help, problem-solving

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⚡
**Quick cheat sheet** for experienced users.
- ✅ One-liner commands
- ✅ Command overview
- ✅ Common issues table
- 📖 **Read this for:** Reminding yourself of commands, quick lookup

---

## 🖱️ Batch Command Files

### [Start_VeilAI_Lite.bat](Start_VeilAI_Lite.bat) 
**Easiest way to launch** (recommended)
- Double-click to run
- Opens browser automatically
- Lightweight version (fast, minimal dependencies)
- **Best for:** Quick testing, trying out protection

### [Start_VeilAI_Full.bat](Start_VeilAI_Full.bat)
**Launch full version with all features**
- Double-click to run
- Opens browser automatically
- Full CLIP model + COCO dataset
- **Best for:** Production use, all features

---

## 📊 Decision Tree

**Which file should I use?**

```
Are you on Windows?
├─ YES → Use Start_VeilAI_Lite.bat ✅ (easiest)
└─ NO  → Use command line (see RUN_VEILAI.md)

Want to run from command line?
├─ YES → Copy commands from QUICK_REFERENCE.md
└─ NO  → Double-click a .bat file

Need help or troubleshooting?
├─ YES → Read RUN_VEILAI.md (detailed guide)
└─ NO  → Use QUICK_REFERENCE.md (quick lookup)

First time using VeilAI?
├─ YES → Start with RUN_VEILAI.md (section 1 & 2)
└─ NO  → Just use Start_VeilAI_Lite.bat
```

---

## ⚡ TL;DR (Super Quick Version)

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
python launch_ui.py --lite
```

Open: http://localhost:7860

Done! 🎉

---

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Nothing happens when I click .bat | Python might not be installed; use command line version |
| "Module not found" error | Run `pip install -r requirements.txt` first |
| Port 7860 already in use | Kill other VeilAI instance or use `--port 7861` |
| Very slow loading | First run of `--full` downloads CLIP; use `--lite` for speed |
| Can't open browser | Manually go to http://localhost:7860 |

**Still stuck?** See the full troubleshooting section in [RUN_VEILAI.md](RUN_VEILAI.md#troubleshooting)

---

## 📋 Command Reference

### Lightweight Version (Recommended)
```bash
python launch_ui.py --lite          # Fastest, minimal dependencies
```

### Full Version
```bash
python launch_ui.py --full          # All features, slower
```

### Other Commands
```bash
python launch_ui.py --demo          # Demo without CLIP
python launch_ui.py --config        # Show current settings
python launch_ui.py --help          # Show all options
```

---

## 🎯 Next Steps After Launching

1. **Upload an image** - JPG, PNG, or GIF work great
2. **Adjust protection strength** - Use the slider (50-80% recommended)
3. **View the result** - Cloaked image appears on right
4. **Download** - Right-click and save to your computer

---

## 📂 File Structure Reference

```
c:\Users\Acer\Documents\Programs\UAP\
├── instructions/              ← You are here
│   ├── README.md             (This file - overview)
│   ├── RUN_VEILAI.md         (Detailed guide)
│   ├── QUICK_REFERENCE.md    (Quick cheat sheet)
│   ├── Start_VeilAI_Lite.bat (Easy Windows launcher)
│   └── Start_VeilAI_Full.bat (Full features launcher)
├── python/
│   ├── launch_ui.py          (Main script - don't edit)
│   ├── gradio_ui_lite.py     (Lightweight UI - don't edit)
│   ├── gradio_ui.py          (Full UI - don't edit)
│   ├── requirements.txt       (Dependencies)
│   ├── UI_README.md
│   ├── QUICK_START.md
│   └── ...
└── README.md                 (Project overview)
```

---

## 🔧 Setup Requirements

- **Python:** 3.8 or higher
- **OS:** Windows 10/11 (also works on Mac/Linux)
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** ~500MB for CLIP model (first run only)
- **Internet:** Required for first `--full` run (downloads CLIP)

---

## 💡 Pro Tips

### Tip 1: Create Desktop Shortcut
Right-click `Start_VeilAI_Lite.bat` → Send to → Desktop

### Tip 2: Pin to Taskbar (Windows)
Run `Start_VeilAI_Lite.bat` → Right-click icon in taskbar → Pin to taskbar

### Tip 3: Different Port
If port 7860 conflicts, run:
```bash
python launch_ui.py --lite --port 7861
```

### Tip 4: No Browser Auto-Open
Add `--no-browser` flag:
```bash
python launch_ui.py --lite --no-browser
```

---

## 📞 Support

- **Can't find Python?** See "First Time Setup" in [RUN_VEILAI.md](RUN_VEILAI.md)
- **Installation issues?** See "Troubleshooting" in [RUN_VEILAI.md](RUN_VEILAI.md#troubleshooting)
- **Command questions?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **General help?** Read [RUN_VEILAI.md](RUN_VEILAI.md) (most comprehensive)

---

**Last Updated:** March 13, 2026  
**VeilAI Version:** 1.0  
**Status:** Ready to use! 🚀
