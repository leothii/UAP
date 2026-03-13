# ⚡ Quick Reference Sheet

## One-Liner Cheat Sheet

```bash
# Lightweight (recommended for testing)
python launch_ui.py --lite

# Full version (with all features)
python launch_ui.py --full

# Show configuration
python launch_ui.py --config
```

---

## Before Running

Make sure you're in the right directory:

```bash
cd c:\Users\Acer\Documents\Programs\UAP\python
```

Or set up PATH so you can run from anywhere.

---

## What to Expect

### Lightweight Version (~2-5 seconds startup)

```
📱 Launching VeilAI - Lightweight Version...
This version has minimal dependencies

Running on local URL: http://127.0.0.1:7860
```

✅ Open browser to: **http://localhost:7860**

### Full Version (~30-60 seconds first time)

```
🚀 Launching VeilAI - Full-Featured Version...
This version requires all dependencies

Loading CLIP model... [████████████] 100%

Running on local URL: http://127.0.0.1:7860
```

✅ Open browser to: **http://localhost:7860**

---

## Commands

| Task | Command |
|------|---------|
| Run lightweight VeilAI | `python launch_ui.py --lite` |
| Run full VeilAI | `python launch_ui.py --full` |
| View config | `python launch_ui.py --config` |
| Show help | `python launch_ui.py --help` |
| Stop VeilAI | `Ctrl + C` (in terminal) |

---

## Ports

- **Default:** http://localhost:7860
- **Alternative:** http://127.0.0.1:7860

If port 7860 is in use, change it:

```bash
python launch_ui.py --lite --port 7861
```

---

## Internet Required?

- **First run (Full version):** Yes, downloads CLIP model (~350MB)
- **Subsequent runs:** No, uses cached model
- **Lightweight version:** Only needs Gradio

---

## File Structure

```
UAP/
├── instructions/              ← You are here
│   ├── RUN_VEILAI.md         (Main guide)
│   └── QUICK_REFERENCE.md    (This file)
├── python/
│   ├── launch_ui.py          (Run this)
│   ├── gradio_ui.py          (Full version)
│   ├── gradio_ui_lite.py     (Lightweight version)
│   └── requirements.txt
└── README.md
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Port already in use | Kill other VeilAI, or use `--port 7861` |
| "No module named 'gradio'" | Run `pip install -r requirements.txt` |
| "CUDA out of memory" | Use `--lite` version or `--cpu` flag |
| Interface won't load | Check http://localhost:7860, try different browser |
| Very slow processing | Switch from `--full` to `--lite` version |

---

## Default Settings

- **Server Port:** 7860
- **Theme:** Professional (dark blues)
- **Protection Strength:** 0-100% (slider)
- **Image Format:** JPG, PNG, GIF supported
- **Max Image Size:** No limit (system dependent)

---

**Pro Tip:** Create a shortcut or batch file to launch with one click! See `RUN_VEILAI.md` for instructions.
