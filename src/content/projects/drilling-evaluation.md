---
title: "Craniotomy Training Evaluation UI"
date: 2026-06-01
draft: false
projectType: "Work"
tags:
- Deep Learning
- Desktop App
- Python
description: "PyQt5 desktop app for neurosurgery craniotomy training — RANSAC homographic perspective correction, async QThread video pipeline, and Grad-CAM heatmaps for explainable AI scoring."
repoURL: "https://github.com/NetsAiims/craniotomyTrainingEvaluation"
---
## TL;DR
A standalone desktop application designed to evaluate the surgical performance of neurosurgeons during craniotomy (skull drilling) training procedures. The system bridges the gap between clinical evaluation and deep learning by providing an intuitive, real-time GUI backed by explainable AI metrics.

## Subjectivity in Surgical Training
Evaluating surgical technique, specifically the precision of bone drilling in neurosurgery, is traditionally highly subjective and relies on senior surgeons eyeballing post-operative results. The technical problem is capturing high-resolution clinical camera feeds without freezing the UI, standardizing the visual perspective of the drilled bone, and using AI to provide an objective score while explicitly proving to the surgeon *why* that score was given.

## The Need for Explainable Clinical AI
In a clinical setting, an AI model that simply outputs a score is a "black box" that doctors will inherently distrust. We needed an application that not only evaluates performance but generates clinical-grade reports with mathematical visual proof. Furthermore, the application had to run flawlessly on standard hospital hardware without requiring technical expertise, forcing a strict separation between the heavy OpenCV/PyTorch backend and the PySide/PyQt frontend.

## Non-Blocking UI and Mathematical Visual Proof

### Non-Blocking Asynchronous UI
The GUI is built in Python using a multi-threaded architecture. To ensure the application remains perfectly responsive while pulling heavy HD camera frames, the video capture loop runs inside a dedicated `QThread`. Frames are emitted to the main UI thread purely via PyQt `Signals`, preventing the dreaded "Application Not Responding" freeze during heavy model inference.

### Geometric Normalization (Homography)
Cameras are rarely mounted at the exact same angle. Before the AI evaluates a drill site, the system uses **RANSAC (Random Sample Consensus)** to calculate a Homography matrix. This mathematically warps and flattens the image to a standardized top-down perspective, ensuring the deep learning model evaluates the actual bone structure and not camera distortion.

### Explainable AI (Grad-CAM)
To solve the "black box" problem, I integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** using the Captum library. During evaluation, the system calculates the gradients of the target class flowing back into the final convolutional layer. This generates a spatial heatmap overlaid on the surgical site, mathematically proving exactly which drill marks influenced the model's final score.

### Automated Clinical Reporting
A dedicated `pdfWriter` module takes the standardized images, the AI scores, and the Grad-CAM heatmaps, compiling them into a formatted PDF report that serves as the official objective record of the trainee's performance.
