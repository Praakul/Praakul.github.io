---
title: "Real-Time Industrial Posture Analysis"
date: 2026-02-15
draft: false
tags: ["PyTorch", "ST-GCN", "MediaPipe", "Computer Vision", "Edge Computing"]
badges: ["PyTorch", "MediaPipe", "ST-GCN", "FastAPI"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul/postureEstimation
---

Manual material handling is still one of the biggest causes of musculoskeletal injuries in factories, and traditional safety protocols just don't give real-time, actionable feedback. To tackle this, I built a production-grade **Industrial IoT system** that combines deterministic bio-mechanical physics with probabilistic deep learning, all running seamlessly across a scalable **edge-cloud architecture**.

---

### How the System is Built

I decoupled the architecture into three distinct layers to keep it fast and scalable:

#### The Edge Client: Processing on the Factory Floor

Running right on the factory floor (using a laptop, NUC, or Jetson), the client handles four sequential transformations on raw camera frames so the heavy video data never has to leave the edge:

1. **Skeleton Extraction:** I use MediaPipe Pose to extract a 33-point 3D skeleton at roughly 30 FPS. I only keep the 17 COCO keypoints (like the nose, shoulders, elbows, hips, etc.) and toss out redundant extremities to keep the dimensionality low.

2. **Temporal Smoothing:** Raw keypoint data can be pretty jittery. So, each of the 17×3 coordinate channels goes through an independent **OneEuroFilter**—an awesome adaptive low-pass filter. It dynamically adjusts its cutoff frequency depending on how fast the person is moving. If a worker is standing still, it smooths hard to eliminate jitter. If they move fast, it opens up to stay responsive. I configured it with `min_cutoff=1.0` and `beta=0.007`.

3. **Geometric Normalization:** Next, I transform the skeleton to make the model invariant to camera position:
   - **Position Invariance:** I shift the hip center to the origin `(0,0,0)`.
   - **View Invariance:** I rotate the skeleton around the Y-axis (gravity) so the hip aligns with the X-axis. This is calculated as `θ = arctan2(dz, dx)` of the hip vector, making the system robust even if the camera is placed diagonally or to the side.
   - **Scale Invariance:** All sets of coordinates are divided by the spine length, normalizing the distance from the camera.

4. **Velocity Computation:** I compute frame-to-frame position changes for each joint, generating 3 extra channels (vx, vy, vz) per keypoint. The final feature vector for each frame comes out to `17 joints × 6 channels = 102 dimensions`.

#### The Inference Server: AI at the Core

The server hosts the heavy lifting—a **6-block ST-GCN** (Spatial-Temporal Graph Convolutional Network) with channel-wise attention.

**A peek into the architecture:**
- **Input shape:** `(Batch, 6, 50, 17)` — 6 channels (position + velocity), 50 frames (about 1.7 seconds of video), and 17 joints.
- **Graph setup:** The human skeleton topology is structured as a 17-node adjacency matrix. For stable gradients, I symmetrically normalized it using `D^{-1/2} A D^{-1/2}`.
- **Inside each ST-GCN block:**
  - A **Graph Convolution** layer uses a `1×1` convolution followed by Einstein summation (`einsum`) to pass messages across joints.
  - A **Temporal Convolution** captures motion patterns across frames, followed by BatchNorm and Dropout.
  - A **Squeeze-and-Excitation (SE) block** acts as channel-wise attention, highlighting critical load-bearing joints like the spine or hips and suppressing peripheral noise.
- **Layers:** The network grows from 6 → 64 → ... → 256 channels, downsampling along the way.
- **Output:** It finishes with a global average pooling and a linear layer predicting 3 classes: Safe, Warning, or Critical.

To tie it together, the client and server communicate instantly over **WebSockets** via FastAPI, passing 50-frame sequences as lightweight JSON. The server returns its safety prediction in milliseconds.

#### The Safety Logic: Making the Call

This isn't just an AI model; there's a deterministic decision engine acting as a safety net:

- **Bio-Mechanical Rules (EAWS/NIOSH):** 
  - Neutral (small flexion): Safe
  - Bent: Warning if stooping (straight knees), Safe if squatting correctly.
  - Strongly Bent: Critical
- **Context Awareness:** Alerts only trigger if the worker's hands are actually below their knees, confirming a lifting motion.
- **Temporal Debouncing:** To prevent annoying alarm fatigue, a `SafetyLogic` class maintains a rolling history. A "CRITICAL" alert only sounds if we get 20+ critical predictions within 30 frames, followed by a cooldown period.

---

### How the Model Was Trained

#### Generating Data Automatically
Manually labeling data is slow, so I built a pipeline (`generate_data.py`) to process raw training videos and automatically annotate them using the EAWS bio-mechanical rules as a weak supervisor. It computes 3D torso and knee angles to automatically assign Safe/Warning/Critical labels, generating perfectly labeled feature vectors for over 58,000 frames.

#### Curating and Augmenting Data
I extracted 50-frame sequences using a sliding window. Because safety is the priority, any sequence with more than 20% "Critical" frames was labeled entirely as Critical.

To make the model tough, I injected **physics-grounded augmentations**:
- **Random Y-axis rotation** to simulate different camera angles.
- **Gaussian jitter** to simulate sensor noise.
- **Limb occlusion**, where I'd randomly zero out leg or arm joints to simulate the partial views you often get in a cluttered factory.

#### Training Setup
- **Optimizer:** SGD with momentum and weight decay.
- **Loss:** Weighted CrossEntropyLoss with label smoothing (to handle class imbalances).
- **Scheduler:** MultiStepLR.

#### The Results
- **Validation Accuracy:** Kept at an intentional 88.24% to prioritize real-world generalization over simply overfitting the training set.
- **Latency:** Less than 50ms end-to-end via WebSocket.
- **Output:** A sleek, color-coded visual feedback system (🟢 Safe, 🟠 Warning, 🔴 Critical).

---

### Tech Stack
`Python` · `PyTorch` · `MediaPipe` · `FastAPI` · `WebSockets` · `PyQt6` · `OpenCV` · `NumPy` · `scikit-learn`