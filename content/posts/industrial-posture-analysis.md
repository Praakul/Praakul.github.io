---
title: "Real-Time Industrial Posture Analysis"
date: 2026-02-15
draft: false
tags: ["PyTorch", "ST-GCN", "MediaPipe", "Computer Vision"]
---

### The Problem
Ergonomic risks in industrial environments lead to long-term injuries. [cite_start]The goal was to build a system capable of classifying complex postures (Stoop vs. Squat) in real-time[cite: 21, 22].

### The Solution & Architecture
[cite_start]I engineered an end-to-end pipeline using **MediaPipe** for skeletal extraction and a **Spatial-Temporal Graph Convolutional Network (ST-GCN)** for action recognition[cite: 22]. 

To handle the lack of labeled industrial data, I built a weak-supervision pipeline utilizing bio-mechanical heuristics. [cite_start]This physics-aware approach allowed me to auto-label over 58,000 frames from raw video inputs[cite: 23].

### Inference Demo
*Real-time skeleton extraction and posture classification:*

<!-- {{< video src="/videos/posture-demo.mp4" >}} -->

### Optimization
[cite_start]To improve the model's focus on critical joints (like the lower back and knees), I integrated **Squeeze-and-Excitation (SE) attention blocks**, which pushed the classification accuracy to **88.2%**[cite: 23].