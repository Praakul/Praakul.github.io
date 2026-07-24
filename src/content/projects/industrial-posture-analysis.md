---
title: Real-Time Industrial Posture Analysis
date: 2026-02-15
draft: false
projectType: "Projects"
tags:
- PyTorch
- ST-GCN
- MediaPipe
- Computer Vision
- Edge Computing
badges:
- PyTorch
- MediaPipe
- ST-GCN
- FastAPI
description: "Edge-cloud IoT system combining ST-GCN neural networks with deterministic EAWS bio-mechanical rules to classify factory worker posture in real-time at <50ms latency."
repoURL: https://github.com/Praakul/postureEstimation
---

## TL;DR
Manual material handling is still one of the biggest causes of musculoskeletal injuries in factories, and traditional safety protocols fail to provide real-time, actionable feedback. I built a production-grade Industrial IoT system that combines deterministic bio-mechanical physics with probabilistic deep learning, running seamlessly across a scalable edge-cloud architecture to monitor and correct worker posture in real-time.

## Mapping 2D Video to 3D Physics
The core problem is mapping raw 2D video into a 3D structural understanding of the human body, and then continuously evaluating that structure against ergonomic safety standards (like EAWS). Relying entirely on deep learning sequence models (like LSTMs on raw frames) is computationally impossible on the edge and highly sensitive to camera placement. The problem must be stripped down to extracting key structural points (joints), mathematically normalizing them against gravity and depth, and predicting safety based strictly on spatial-temporal geometry.

## Edge Constraints and Camera Invariance
The system was designed for the factory floor, which enforces severe constraints: computation must be split (Edge vs. Cloud) to avoid transmitting heavy video feeds, and the model must be completely invariant to where the camera is physically mounted. We chose a Spatial-Temporal Graph Convolutional Network (ST-GCN) over standard CNNs because human movement is fundamentally a graph (joints connected by bones). To achieve functional safety standards (ISO 26262), we could not rely purely on a "black box" AI; the system required a deterministic fallback engine (calculating exact joint angles) to act as a hard constraint on the neural network.

## The ST-GCN and Fallback Engine

The system is decoupled into an Edge Client and a Cloud Inference Server communicating via FastAPI WebSockets for <50ms latency.

### The Edge Pipeline (Transforming Physics)
The Edge Client (running on a Jetson or NUC) never transmits raw video. It extracts 17 COCO keypoints using MediaPipe and applies four critical mathematical transformations:
1. **Temporal Smoothing**: Applies a 1-Euro Filter. Unlike a basic moving average, this adaptive low-pass filter dynamically changes its cutoff frequency based on movement speed to eliminate jitter without introducing lag.
2. **Geometric Normalization**: Employs Geometric Algebra to shift the hip center to the origin (0,0,0) and rotates the skeleton around the Y-axis (gravity) so the hip aligns with the X-axis (θ = arctan2(dz, dx)). This makes the system perfectly robust to off-angle camera mounts.
3. **Scale Invariance**: Normalizes all coordinates by spine length to counter the lack of true metric scale from single-lens RGB cameras.
4. **Kinematics**: Explicitly calculates 1st-Order derivatives (Vx, Vy, Vz) to feed the network pre-calculated velocity tensors, rather than forcing the network to infer them.

### The Inference Server (ST-GCN)
The server processes sequences of 50 frames via a custom **6-block ST-GCN**. 
* **The Graph Structure**: The human skeleton is defined as an adjacency matrix, symmetrically normalized (D^(-1/2) A D^(-1/2)) for stable gradients.
* **Squeeze-and-Excitation (SE) Attention**: To handle occlusion and noisy data (e.g., when a worker carries a large box), SE blocks are integrated directly into the spatio-temporal layers. These mathematically "excite" critical load-bearing joints (spine, hips) and "squeeze" out peripheral noise during high-velocity lifts.

### The Real-World Safety Engine
To prevent alarm fatigue, a temporal debouncing logic engine tracks a rolling history. A "CRITICAL" alert only sounds if 20+ critical predictions occur within a 30-frame window. If the ST-GCN drops confidence, the system seamlessly hands control to the Bio-Mechanical Fallback Engine to deterministically calculate Euclidean torso and knee angles based on EAWS standards.