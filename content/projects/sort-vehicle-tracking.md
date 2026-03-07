---
title: "Multi-Vehicle Tracking with SORT"
date: 2025-11-20
draft: false
tags: ["Computer Vision", "Kalman Filter", "FastAPI", "Docker", "CI/CD"]
badges: ["OpenCV", "FastAPI", "Docker", "YOLOv8"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul/multipleVehicleTracking
  - icon: fas fa-globe
    url: https://prajwalkulkarni-vehicletracking.hf.space/docs
---

A from-scratch implementation of the **SORT (Simple Online and Realtime Tracking)** algorithm for multi-vehicle tracking. Every component — the Kalman Filter, the Hungarian assignment solver, and the track lifecycle manager — is written from first principles, not imported from a library. Deployed as a production FastAPI service on Hugging Face Spaces.

---

## The Kalman Filter — From First Principles

At the core of each tracked vehicle is a **6-dimensional Kalman Filter** implementing a constant-velocity motion model:

### State Vector
```
x = [cx, cy, w, h, vx, vy]
```
Where `cx, cy` = bounding box center, `w, h` = width/height, `vx, vy` = center velocity.

### State Transition (Predict Step)
The transition matrix `F` encodes the constant velocity assumption:
```
cx' = cx + vx·dt    (position updates by velocity)
cy' = cy + vy·dt
w'  = w              (size stays constant between frames)
h'  = h
vx' = vx             (velocity stays constant)
vy' = vy
```
State covariance propagation: `P' = F·P·Fᵀ + Q`

### Measurement Model (Update Step)
Only position and size are directly observed from YOLO detections (4D measurement = `[cx, cy, w, h]`). The velocities are estimated indirectly through the filter's internal state.

The update follows the standard Kalman equations:
1. **Innovation:** `y = measurement - H·state` (how wrong was our prediction?)
2. **Innovation covariance:** `S = H·P·Hᵀ + R`
3. **Kalman Gain:** `K = P·Hᵀ·S⁻¹` (how much to trust measurement vs. prediction)
4. **State update:** `state = state + K·y`
5. **Covariance update:** `P = (I - K·H)·P`

The noise covariances are tuned for traffic video: process noise `Q` uses `q_pos=0.1` (positions are predictable) and `q_vel=1.0` (accelerations happen). Measurement noise `R = 5.0·I` reflects YOLOv8's detection precision.

---

## The SORT Pipeline

Each frame passes through four stages:

### 1. Detection
A pre-trained **YOLOv8n** model processes the frame, filtering for COCO vehicle classes only: cars (2), buses (5), trucks (7). Detections below the confidence threshold are discarded.

### 2. Prediction
Every existing track's Kalman Filter predicts its expected position in the current frame. The track's `time_since_update` counter increments.

### 3. Association — Hungarian Algorithm
The core matching problem: given N predicted track positions and M new detections, find the optimal assignment.

- A **cost matrix** of shape `(N_tracks, M_detections)` is computed, where each entry is `1 - IoU(predicted_bbox, detected_bbox)`
- The **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) solves this as a minimum-cost assignment problem
- Matches with cost > 0.7 (i.e., IoU < 0.3) are rejected as spurious

### 4. Track Lifecycle
- **Matched detections** → update the corresponding Kalman Filter (state correction), reset `time_since_update` to 0
- **Unmatched detections** → initialize a new `Track` with a fresh Kalman Filter, assign an incrementing track ID
- **Unmatched tracks** → retain if `time_since_update ≤ 5` (vehicle temporarily occluded), otherwise delete

The output is a list of active tracks — each with a stable ID, a predicted bounding box, and a Kalman-smoothed trajectory.

---

## Deployment

The full pipeline is containerized and deployed as a production API:

- **FastAPI** handles async file upload, video processing, and file download
- **Docker** multi-stage build for reproducible deployment
- **Hugging Face Spaces** with the Docker SDK for public hosting
- **GitHub Actions** CI/CD pipeline triggers automatic redeployment on push to `main`
- **Interactive API docs** at `/docs` — upload any video, download the tracked output

### Usage
**Via API:**
```bash
curl -X POST "https://prajwalkulkarni-vehicletracking.hf.space/track_video/" \
     -F "file=@your_video.mp4" -o "tracked_output.mp4"
```

Or visit the [live interactive docs](https://prajwalkulkarni-vehicletracking.hf.space/docs) to try it directly.

---

## Stack
`Python` · `NumPy` · `SciPy` · `OpenCV` · `Ultralytics (YOLOv8)` · `FastAPI` · `Docker` · `GitHub Actions`