
Mobile-Based Cloaking of Proprietary Images against Unauthorized AI Training

Researcher:
Capayan, Quinjie Benedict
Chua, Ralph Martin
Diana, Gabriel
Libuna, Donjie c.

Institution: West Visayas State University

Degree: Bachelor of Science in Computer Science (Major in AI)

📌 Project Overview
This repository contains the implementation of a defensive mechanism designed to protect proprietary images from being harvested for AI model training. The system generates Universal Adversarial Perturbations (UAPs) that semantic-aligning models, such as CLIP (ViT-B/32), cannot correctly interpret.

The primary goal is to deploy these protections via a mobile application, allowing creators to "cloak" their images using hardware-accelerated Alpha Blending before digital publication.

🚀 Key Technical Features
Semantic Disruption: Targets the vision-language alignment of the CLIP framework to minimize Cosine Similarity between image features and text descriptions.

Mobile-Optimized Optimization: The training loop incorporates a dedicated Alpha Blending parameter (α=0.7) to ensure the perturbation remains effective when rendered at partial opacity on a mobile device.

High-Fidelity Constraints: Generation is bounded to ensure a Structural Similarity Index (SSIM) > 0.90, preserving the aesthetic value of the original art while maintaining high protection levels.

Resource-Efficient Pipeline: Utilizes a custom MS-COCO 2017 data loader designed for mini-batching (1,000 images), enabling optimization on consumer-grade CPU hardware.

📁 Repository Structure
```
UAP/
├── README.md                       # Project documentation
├── data/                           # Dataset and results storage
│   ├── MS-COCO/                    # MS-COCO 2017 dataset
│   │   └── val2017/                # Validation images (5,000 images)
│   └── results/                    # Generated UAPs and visualizations
├── mobile-assets/                  # Final mobile-ready PNG overlays
├── progress_reports/               # Research documentation and reports
├── python/                         # Core research scripts
│   ├── clip_integration.py         # Standardized wrapper for CLIP ViT-B/32
│   ├── clip_uap_generator.py       # Main UAP engine (Moosavi-Dezfooli et al. base)
│   ├── coco_loader.py              # Memory-efficient MS-COCO pipeline
│   ├── export_mobile_assets.py     # Export perturbations for mobile deployment
│   ├── fidelity_validator.py       # SSIM, PSNR, and Similarity drop validation
│   └── requirements.txt            # Python dependencies
└── universal-master/               # Original DeepFool UAP research code
    ├── README.md                   # Original repository documentation
    ├── matlab/                     # MATLAB implementation
    │   ├── universal_perturbation.m
    │   ├── demo_caffe.m
    │   └── data/                   # Pretrained model definitions
    ├── precomputed/                # Pre-generated UAPs for ImageNet
    │   ├── CaffeNet.mat
    │   ├── GoogLeNet.mat
    │   ├── ResNet-152.mat
    │   └── VGG-*.mat
    └── python/                     # Python implementation
        ├── universal_pert.py       # Original UAP algorithm
        ├── deepfool.py             # DeepFool base algorithm
        └── data/                   # Labels and precomputed data
```


🛠️ Installation & Setup
To replicate the research environment, follow these steps to install the necessary dependencies:

Clone the repository:

Bash
git clone https://github.com/leothii/UAP.git
cd UAP/python
Create a virtual environment (Recommended):

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
Ensure you have pip updated before installing the requirements.

Bash
pip install -r requirements.txt
Note: This project requires PyTorch for the CLIP model wrapper and TensorFlow for the core UAP generation engine.

🛠️ Execution Pipeline
To ensure reproducible results for progress reports, run scripts in the following order:

Environment Test: python clip_integration.py (Verify hardware and model loading).

Data Check: python coco_loader.py (Verify MS-COCO directory access).

UAP Generation: python clip_uap_generator.py (Generate the "Universal Cloak").

Validation: python fidelity_validator.py (Calculate SSIM, PSNR, and Fooling Rate).

🤝 Contributing
This repository is primarily for academic research purposes at West Visayas State University. However, contributions that improve the efficiency of the UAP generation or the mobile deployment pipeline are welcome.

Reporting Bugs: Please open an issue if you encounter errors in the MS-COCO data loader or CLIP integration.

Feature Requests: Suggestions for improving the Alpha Blending math or SSIM validation logic are highly appreciated.

Pull Requests: Ensure all code follows the "Clean Production" modular structure established in the python/ directory.

⚖️ License & Ethical Use
Academic License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code for research and educational purposes, provided that proper credit is given to the author and the institution.

Ethical AI Disclaimer
The goal of this research is to empower creators and protect intellectual property. This tool should not be used to disrupt legitimate, authorized AI research or to facilitate malicious attacks on machine learning systems. It is strictly a defense against automated semantic feature extraction and unauthorized fine-tuning.

📊 Experimental Validation
Final effectiveness is measured by a controlled study:

Control: A Stable Diffusion XL (SDXL) model fine-tuned via LoRA on clean images.

Experimental: An identical SDXL model fine-tuned on "cloaked" images.

Success Criteria: The Experimental group must fail to replicate the proprietary style or content of the dataset.

📜 Academic References
Moosavi-Dezfooli, S. et al. (2017): Universal adversarial perturbations.

Radford, A. et al. (2021): Learning Transferable Visual Models from Natural Language Supervision (CLIP).

Wang, Z. et al. (2004): Image quality assessment: from error visibility to structural similarity (SSIM).