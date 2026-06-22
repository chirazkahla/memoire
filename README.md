# Advanced Deep Learning and Imaging Techniques for EarlyDetection and Progression Analysis of Knee Osteoarthritis

## Overview

This repository contains the Master's thesis entitled:

**"Advanced Deep Learning and Imaging Techniques for EarlyDetection and Progression Analysis of Knee Osteoarthritis"**

The project proposes a computer-aided diagnosis framework for the automated analysis of Knee Osteoarthritis (KOA) using Magnetic Resonance Imaging (MRI) and advanced deep learning techniques.

## Objectives

* Early detection of Knee Osteoarthritis.
* Automated multi-class anatomical segmentation of knee structures.
* Extraction of quantitative imaging biomarkers.
* Prediction and analysis of disease progression.
* Development of an AI-assisted diagnostic framework for clinical decision support.

## Dataset

**OAIZIB-CM Dataset**

* MRI-based knee osteoarthritis dataset.
* Multiple anatomical structures including cartilage and menisci.
* Used for segmentation, feature extraction, and disease progression analysis.

## Methodology

### 1. MRI Preprocessing

* Wavelet Denoising
* Restormer Image Enhancement
* Gamma Correction
* CLAHE Contrast Enhancement
* ROI Cropping

### 2. Multi-Class Segmentation

Deep learning architectures:

* UNet
* Swin UNETR

Target structures:

* Femoral Cartilage
* Medial Tibial Cartilage
* Lateral Tibial Cartilage
* Medial Meniscus
* Lateral Meniscus

### 3. Feature Extraction

* Intensity Features
* Texture Features 
* Shape Features
* Deep Features

### 4. Classification and Progression Analysis

Machine learning models were employed to analyze disease severity and progression using extracted imaging biomarkers.

## Main Technologies

* Python
* PyTorch
* MONAI
* NumPy
* OpenCV
* Scikit-Learn
* NiBabel
* Matplotlib

## Results

The proposed framework achieved strong segmentation performance, with a Dice score exceeding 0.97 for the best-performing model.

The study demonstrates the potential of deep learning and MRI-based biomarkers for accurate knee osteoarthritis assessment and progression monitoring.

## Repository Contents

* Master's Thesis PDF
* Experimental Results
* Source Code (to be added)
* Documentation

## Author

**Chiraz Kahla**

Master's Degree in Artificial Intelligence

University of El Oued, Algeria

## Citation

If you use this work, please cite the corresponding Master's thesis.

© 2026 Chiraz Kahla. All Rights Reserved.
