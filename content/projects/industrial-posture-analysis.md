---
title: "Real-Time Industrial Posture Analysis"
date: 2026-02-15
draft: false
tags: ["PyTorch", "ST-GCN", "MediaPipe", "Computer Vision", "Edge Computing"]
badges: ["PyTorch", "MediaPipe", "ST-GCN", "FastAPI"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul/pose
---

Manual material handling remains the primary cause of musculoskeletal disorders in industrial settings. Standard safety protocols fail to provide real-time, actionable feedback. This project implements a production-grade **Industrial IoT system** using hybrid intelligence — combining deterministic bio-mechanical physics with probabilistic deep learning — distributed across a scalable **edge-cloud architecture**.

---

## System Architecture

The system is decoupled into three distinct layers:

### The Edge Client — Signal Processing Pipeline

Running on factory floor hardware (laptop/NUC/Jetson), the client performs four sequential transformations on raw camera frames before data ever leaves the edge:

1. **Skeleton Extraction:** MediaPipe Pose extracts a 33-point 3D skeleton at ~30 FPS. Only 17 COCO keypoints are retained (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) — discarding redundant extremity points to reduce dimensionality.

2. **Temporal Smoothing:** Raw keypoints are noisy. Each of the 17×3 coordinate channels passes through an independent **OneEuroFilter** — an adaptive low-pass filter that dynamically adjusts its cutoff frequency based on signal velocity. When a worker is standing still, the filter aggressively smooths (eliminating jitter). During fast movements, the cutoff rises automatically to preserve responsiveness. This is implement with a configurable `min_cutoff=1.0` and `beta=0.007`.

3. **Geometric Normalization:** The skeleton is transformed to achieve three invariances:
   - **Position Invariance:** The hip center is translated to the origin `(0,0,0)`
   - **View Invariance:** The skeleton is rotated around the Y-axis (gravity) so the hip vector aligns with the global X-axis. This is computed as `θ = arctan2(dz, dx)` of the hip vector, making the model robust to diagonal and side camera angles
   - **Scale Invariance:** All coordinates are divided by the spine length (shoulder center to hip center distance), normalizing distance from camera

4. **Velocity Computation:** Frame-to-frame position deltas are computed per joint, producing 3 additional channels (vx, vy, vz) per keypoint. The final per-frame feature vector is `17 joints × 6 channels = 102 dimensions`.

### The Inference Server — Spatial-Temporal Graph Convolution

The server hosts the core AI model — a **6-block ST-GCN** (Spatial-Temporal Graph Convolutional Network) with channel-wise attention:

**Architecture Details:**
- **Input tensor shape:** `(Batch, 6, 50, 17)` — 6 channels (position + velocity), 50 frames (~1.7 seconds at 30fps), 17 joints
- **Graph definition:** A 17-node adjacency matrix encoding the human skeleton topology (16 edges: nose-eyes, shoulders-elbows-wrists, hips-knees-ankles, cross-body connections). The adjacency matrix is **symmetrically normalized** using `D^{-1/2} A D^{-1/2}` for stable gradient flow
- **Each ST-GCN block** contains:
  - A **Graph Convolution** layer using `1×1` convolution followed by Einstein summation (`einsum('nctv,vw->nctw')`) against the adjacency matrix for spatial message passing across joints
  - A **Temporal Convolution** with kernel size `(9,1)` for capturing motion patterns across frames, followed by BatchNorm and Dropout
  - A **Squeeze-and-Excitation (SE) block** — channel-wise attention that globally average-pools spatial features, passes through a bottleneck FC layer (reduction=16), and produces per-channel importance weights via sigmoid. This allows the network to dynamically emphasize load-bearing joints (spine, hips) while suppressing peripheral noise
- **Channel progression:** 6 → 64 → 64 → 128 → 128 → 256 → 256, with stride-2 temporal downsampling at blocks 3 and 5
- **Classification head:** Global average pooling → Linear(256, 3) → 3 classes (Safe, Warning, Critical)

Communication between client and server uses **WebSockets** via FastAPI, transmitting 50-frame skeleton sequences as JSON. The server processes each sequence with `torch.no_grad()` and returns the prediction class with softmax confidence.

### The Safety Logic — Context-Aware Decision Engine

Not a model — a deterministic decision layer combining multiple information sources:

- **EAWS/NIOSH Bio-Mechanical Rules:**
  - Neutral (0°–20° flexion → torso angle 160°–180°): Safe
  - Bent (20°–60° flexion → angle 120°–160°): Warning if stooping (knees straight, angle > 150°), Safe if squatting (knees bent)
  - Strongly Bent (>60° flexion → angle < 120°): Critical
- **Lifting Context:** Alerts only trigger when wrist Y-coordinate exceeds knee Y-coordinate (hands below knees = lifting context)
- **Temporal Debouncing:** A `SafetyLogic` class maintains a 60-frame rolling history. A "CRITICAL" alert requires 20+ critical predictions in the last 30 frames. After triggering, a 3-second cooldown prevents alarm fatigue. Warning alerts use a longer 5-second cooldown

---

## Training Pipeline

### Data Generation — Weak Supervision at Scale
Instead of manual labeling, the `generate_data.py` pipeline processes raw training videos using the EAWS bio-mechanical labeling function as an automatic annotator:
- Every 3rd frame is processed (SKIP=3) for efficiency
- The `get_bio_mechanical_label()` function computes 3D torso and knee angles from raw landmarks to assign Safe(0)/Warning(1)/Critical(2) labels
- Velocity features are computed as frame-to-frame deltas on the normalized skeleton
- Total output: normalized (102-dim) feature vectors + auto-generated labels for 58,000+ frames

### Dataset & Augmentation
50-frame sequences are extracted with a sliding window (stride=20) from the flat frame array. Each sequence gets a **majority-vote label** with critical class bias — if >20% of frames in a window are "Critical," the entire sequence is labeled Critical (safety-first approach).

**Physics-grounded augmentation** applied during training:
- **Random Y-axis rotation** (±0.3 rad) applied to both position and velocity channels — simulating different camera installation angles
- **Gaussian jitter** (σ=0.002) on position channels
- **Limb occlusion** — 20% chance of zeroing out all leg joints, 10% chance of zeroing one arm (simulating partial view blockage, common in industrial settings)

### Training Configuration
- **Optimizer:** SGD with momentum 0.9 and weight decay
- **Loss:** Weighted CrossEntropyLoss with label smoothing. Class weights are computed dynamically from the training set distribution using inverse frequency weighting
- **Scheduler:** MultiStepLR with configurable milestones
- **Early stopping** with patience counter
- **Metrics:** Accuracy, weighted Precision/Recall/F1 (sklearn), confusion matrix saved at each best-model epoch

### Results
- **Validation Accuracy:** 88.24% — intentionally regularized to prioritize true generalization over overfitting to training data
- **End-to-End Latency:** <50ms via WebSocket
- **Color-coded visual feedback:** 🟢 Green (Safe) → 🟠 Orange (Warning) → 🔴 Red (Critical)

---

## Stack
`Python` · `PyTorch` · `MediaPipe` · `FastAPI` · `WebSockets` · `PyQt6` · `OpenCV` · `NumPy` · `scikit-learn`