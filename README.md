# Lumbar Spine Degenerative Classification using Ultralytics YOLO11

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated deep learning pipeline for detecting, localizing, and grading the severity of lumbar spine degeneration from multi-plane MRI scans, developed for the **RSNA 2024 Lumbar Spine Degenerative Classification Challenge**.

---

## 📌 Overview & Clinical Significance

Low back pain is the leading cause of disability worldwide. Magnetic Resonance Imaging (MRI) is the gold-standard diagnostic modality, but manually reviewing multi-slice volumetric MRI scans across multiple series is time-intensive and subject to inter-observer variability.

This project formulates lumbar spine degeneration analysis using state-of-the-art computer vision models to detect and grade severity (**Normal/Mild**, **Moderate**, or **Severe**) across five lumbar levels ($L_1/L_2$, $L_2/L_3$, $L_3/L_4$, $L_4/L_5$, $L_5/S_1$) for three critical pathologies:
1. **Spinal Canal Stenosis (SCS)** — Imaged via **Sagittal T2 / STIR** series.
2. **Neural Foraminal Narrowing (NFN)** — Imaged via **Sagittal T1** series.
3. **Subarticular Stenosis (SS)** — Imaged via **Axial T2** series.

---

## 🚀 Methodology & Model Architecture

* **Object Detection Formulation:** Formulates anatomical localization and severity grading using **Ultralytics YOLO11** (`yolo11s.pt`), leveraging its improved **C3k2** feature extractor and **C2PSA** (Cross-Stage Partial with Spatial Attention) modules for enhanced fine-grained lesion localization.
* **Isolated Pathological Pipelines:** Dedicated models and YAML configurations for each anatomical condition to prevent gradient interference across differing MRI sequences.
* **Medical DICOM Preprocessing:** Converts raw 16-bit DICOM pixel arrays into normalized, standard-contrast images and maps anatomical landmark coordinates to normalized YOLO bounding boxes.
* **Experiment Tracking:** Integrated with **Comet ML** and **Weights & Biases** for hyperparameter logging, real-time loss tracking, and validation visualization.

---

## 📁 Repository Structure

```
Lumbar-Spine-Degenerative-Classification-using-YOLO/
├── README.md                      # Project documentation and showcase
├── requirements.txt               # Pinned Python dependencies (Ultralytics >= 8.3.0)
├── .gitignore                     # Ignores model weights, raw data, and credentials
├── .env.example                   # Environment configuration template
│
├── configs/                       # Dedicated YOLO11 dataset configurations
│   ├── yolo_nfn.yaml              # Neural Foraminal Narrowing (30 classes)
│   ├── yolo_scs.yaml              # Spinal Canal Stenosis (15 classes)
│   └── yolo_ss.yaml               # Subarticular Stenosis (30 classes)
│
├── lsdc-train-yolo-nfn.ipynb      # Training pipeline for Neural Foraminal Narrowing (YOLO11)
├── lsdc-train-yolo-scs.ipynb      # Training pipeline for Spinal Canal Stenosis (YOLO11)
└── lsdc-train-yolo-ss.ipynb       # Training pipeline for Subarticular Stenosis (YOLO11)
```

---

## 🛠️ Getting Started

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Sentientbee/Lumbar-Spine-Degenerative-Classification-using-YOLO.git
cd Lumbar-Spine-Degenerative-Classification-using-YOLO

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment & API Keys
Copy `.env.example` to `.env` and insert your credentials:
```bash
cp .env.example .env
```

```env
# Comet ML (Optional)
COMET_API_KEY=your_api_key_here
COMET_PROJECT_NAME=lsdc-sip

# Weights & Biases (Optional)
WANDB_API_KEY=your_api_key_here
```

### 3. Dataset Configuration
Datasets follow standard Ultralytics YOLO structure (`images/train`, `images/val`, `labels/train`, `labels/val`). Ensure your dataset path is configured in the corresponding YAML file under `configs/`:
* `configs/yolo_nfn.yaml`
* `configs/yolo_scs.yaml`
* `configs/yolo_ss.yaml`

---

## 📊 Models & Class Breakdown

| Condition | MRI Plane | Classes | YOLO11 Model | Target Loss / Task |
| :--- | :--- | :---: | :---: | :--- |
| **Spinal Canal Stenosis** | Sagittal T2 | 15 | `yolo11s.pt` | Bounding Box + 15-class classification |
| **Neural Foraminal Narrowing** | Sagittal T1 | 30 | `yolo11s.pt` | Bounding Box + 30-class classification |
| **Subarticular Stenosis** | Axial T2 | 30 | `yolo11s.pt` | Bounding Box + 30-class classification |

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
