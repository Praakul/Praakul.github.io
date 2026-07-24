---
title: Multi-Vehicle Tracking with SORT
date: 2025-11-20
draft: false
projectType: "Work"
tags:
- Computer Vision
- Kalman Filter
- FastAPI
- Docker
- CI/CD
badges:
- OpenCV
- FastAPI
- Docker
- YOLOv8
description: "From-scratch SORT tracker: Kalman Filter state-space prediction + Hungarian algorithm cost-matrix matching, deployed as a live FastAPI microservice on Hugging Face Spaces."
repoURL: https://github.com/Praakul/multipleVehicleTracking
demoURL: https://huggingface.co/spaces/PrajwalKulkarni/vehicleTracking
---

## TL;DR
A completely from-scratch implementation of the SORT (Simple Online and Realtime Tracking) algorithm to track multiple vehicles in traffic videos. Instead of importing high-level tracking libraries, I engineered the core mathematics (Kalman Filters, Hungarian algorithm) from the ground up to deeply understand the state-space models before packaging it as a highly scalable FastAPI microservice.

## The Temporal Disconnect in Object Detection
Tracking objects across video frames is not just about detecting them; it is about associating a detection in Frame A with the correct detection in Frame B, even if the object is temporarily occluded. The core problem is predicting an object's future position based on its past velocity, and mathematically determining the "cost" of assigning a new detection to that prediction.

## Physics over Recurrent Networks
Deep learning (like YOLOv8) is excellent at spatial understanding (finding the car in a single frame), but it has no temporal memory. Passing entire video sequences into memory-heavy Recurrent Neural Networks (RNNs) for tracking is computationally expensive and overkill for standard traffic monitoring. We needed a purely mathematical, physics-based approach to connect the temporal dots between independent YOLO detections without slowing down inference.

## Engineering the SORT Pipeline

The system relies on a two-stage spatial-temporal pipeline deployed via Docker and FastAPI.

### 1. Spatial Detection (YOLOv8)
A quantized YOLOv8 model extracts the bounding boxes `[cx, cy, w, h]` of vehicles in every frame. We strictly filter for COCO vehicle classes and threshold out low-confidence noise.

### 2. Temporal Prediction (Kalman Filter)
Every tracked vehicle maintains an independent **6-dimensional Kalman Filter** modeling constant-velocity motion. 
The internal state is x = [cx, cy, w, h, vx, vy].
- **Prediction Phase**: P' = F·P·Fᵀ + Q. Before we even look at the next frame's YOLO detections, the filter uses physics (cx' = cx + vx·dt) to predict where the car *should* be.
- **Update Phase**: When YOLO provides the new measurement, we calculate the Kalman Gain (K = P·Hᵀ·S⁻¹). This acts as a mathematical confidence score, dynamically weighting whether to trust our physics prediction or the new YOLO measurement more.

### 3. Track Association (Hungarian Algorithm)
To solve the assignment problem between *predicted* bounding boxes and *actual* YOLO detections, I compute a cost matrix based on 1 - IoU (Intersection over Union). The Hungarian Algorithm (scipy.optimize.linear_sum_assignment) then solves this matrix in polynomial time to find the mathematically optimal pairing.

### Deployment Architecture
The entire physics engine and ML pipeline is containerized using a multi-stage Docker build, exposed via async FastAPI endpoints, and hosted on Hugging Face Spaces with a CI/CD pipeline running through GitHub Actions.