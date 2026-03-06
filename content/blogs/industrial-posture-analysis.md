---
title: "Real-Time Industrial Posture Analysis"
date: 2026-02-15
draft: false
tags: ["PyTorch", "ST-GCN", "MediaPipe", "Computer Vision"]
---

### The Problem
Ergonomic risks in industrial environments lead to long-term injuries. The goal was to build a system capable of classifying complex postures (Stoop vs. Squat) in real-time.

### The Solution & Architecture
I engineered an end-to-end pipeline using **MediaPipe** for skeletal extraction and a **Spatial-Temporal Graph Convolutional Network (ST-GCN)** for action recognition.

To handle the lack of labeled industrial data, I built a weak-supervision pipeline utilizing bio-mechanical heuristics. This physics-aware approach allowed me to auto-label over 58,000 frames from raw video inputs.

### Inference Demo
*Real-time skeleton extraction and posture classification:*



### Optimization
To improve the model's focus on critical joints (like the lower back and knees), I integrated **Squeeze-and-Excitation (SE) attention blocks**, which pushed the classification accuracy to **88.2%**.