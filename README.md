## Overview
This repository contains the training pipelines for the **RSNA 2024 Lumbar Spine Degenerative Classification** challenge. The objective is to aid in the diagnosis of lower back pain by automatically detecting and grading the severity (Normal/Mild, Moderate, or Severe) of three specific conditions: Spinal Canal Stenosis, Neural Foraminal Narrowing, and Subarticular Stenosis.

## Methodology & Tech Stack
* **Data Preprocessing:** Read and converted raw medical DICOM files into normalized images, mapping complex anatomical coordinates to YOLO-formatted bounding boxes.
* **Model Architecture:** Trained separate **Ultralytics YOLOv8** object detection models for different anatomical planes/conditions to maximize localization accuracy and classification recall.
* **Experiment Tracking:** Integrated **Comet ML** to log hyperparameters, track training metrics (Precision, Recall, mAP50, mAP50-95), and visualize validation predictions in real-time.
* **Tools Used:** Python, PyTorch, Ultralytics YOLO, OpenCV, Pydicom, and Pandas.

## Repository Structure
* `lsdc-train-yolo-ss-*.ipynb`: Training notebooks focused on Subarticular Stenosis detection.
* `lsdc-train-yolo-nfn-*.ipynb`: Training notebooks focused on Neural Foraminal Narrowing detection.
* `lsdc-train-yolo-scs-*.ipynb`: Training notebooks focused on Spinal Canal Stenosis detection.
