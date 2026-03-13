"""
Launch script for Gradio UI
Run different versions of the UI based on your needs
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure python directory is in path
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))
os.chdir(str(python_dir))

from ui_config import print_config


def launch_full_ui():
    """Launch the full-featured Gradio UI"""
    print("\n🚀 Launching VeilAI - Full-Featured Version...")
    print("This version requires all dependencies installed")
    print("Make sure you have: torch, clip, torchvision installed\n")
    
    try:
        from gradio_ui import create_gradio_interface
        app = create_gradio_interface()
        app.launch()
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("📦 Install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


def launch_lite_ui():
    """Launch the lightweight UI (works with pre-generated UAPs)"""
    print("\n📱 Launching VeilAI - Lightweight Version...")
    print("This version has minimal dependencies\n")
    
    try:
        from gradio_ui_lite import create_professional_interface
        app = create_professional_interface()
        app.launch()
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("📦 Install gradio:")
        print("   pip install gradio")
        sys.exit(1)


def launch_demo_ui():
    """Launch a demo UI that doesn't require any images or models"""
    print("\n🎬 Launching Demo/Test UI...")
    print("This is a demo version for testing the UI without dependencies\n")
    
    import gradio as gr
    
    with gr.Blocks(title="UAP Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # UAP Image Cloaking - Demo Mode
        
        **This is a demonstration of the UI. No actual UAP processing is happening.**
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image")
                alpha = gr.Slider(0, 1, 0.7, step=0.05)
                btn = gr.Button("Apply Protection")
            
            with gr.Column():
                image_output = gr.Image(label="Result")
                status = gr.Textbox(label="Status")
        
        def demo_apply(img, alpha_val):
            if img is None:
                return None, "No image uploaded"
            return img, f"✓ Demo - Alpha: {alpha_val:.2f}"
        
        btn.click(demo_apply, [image_input, alpha], [image_output, status])
    
    app.launch()


def show_help():
    """Show help information"""
    help_text = """
    UAP Gradio UI - Launch Script
    
    USAGE:
        python launch_ui.py [OPTIONS]
    
    OPTIONS:
        --full, -f          Launch full-featured UI (requires all dependencies)
        --lite, -l          Launch lightweight UI (minimal dependencies)
        --demo, -d          Launch demo UI (no dependencies, testing only)
        --config, -c        Print configuration and exit
        --help, -h          Show this help message
    
    EXAMPLES:
        python launch_ui.py --full
        python launch_ui.py --lite
        python launch_ui.py --demo
    
    RECOMMENDATIONS:
        - First-time setup: run 'pip install -r requirements.txt'
        - Daily use: python launch_ui.py --full
        - Quick testing: python launch_ui.py --lite
        - No dependencies: python launch_ui.py --demo
    """
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Launch UAP Gradio UI",
        add_help=False
    )
    
    parser.add_argument("--full", "-f", action="store_true", 
                       help="Launch full-featured UI")
    parser.add_argument("--lite", "-l", action="store_true",
                       help="Launch lightweight UI")
    parser.add_argument("--demo", "-d", action="store_true",
                       help="Launch demo UI")
    parser.add_argument("--config", "-c", action="store_true",
                       help="Print config and exit")
    parser.add_argument("--help", "-h", action="store_true",
                       help="Show help")
    
    args = parser.parse_args()
    
    # Handle help
    if args.help:
        show_help()
        return
    
    # Handle config
    if args.config:
        print_config()
        return
    
    # Default to full if no option specified
    if not (args.full or args.lite or args.demo):
        print("No option specified. Use --help for usage info")
        print("Defaulting to full UI...\n")
        args.full = True
    
    # Launch the appropriate UI
    try:
        if args.full:
            launch_full_ui()
        elif args.lite:
            launch_lite_ui()
        elif args.demo:
            launch_demo_ui()
    
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
