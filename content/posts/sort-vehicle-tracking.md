---
title: "Real-Time Multi-Vehicle Tracking (SORT)"
date: 2025-11-20
draft: false
tags: ["Computer Vision", "FastAPI", "Docker", "CI/CD"]
---

### System Architecture
I engineered a real-time vehicle tracking system implementing the Simple Online and Realtime Tracking (SORT) algorithm. 

* **State Estimation:** Utilized Kalman Filters to predict the future positions of tracked vehicles.
* **Track Assignment:** Implemented the Hungarian algorithm using Intersection over Union (IoU) metrics to optimally assign bounding boxes across sequential frames.

### Deployment & MLOps
This wasn't just a local script; I built it to be production-grade:
1. **API Layer:** Wrapped the inference engine in a FastAPI service.
2. **Containerization:** Packaged the entire environment using Docker to ensure dependency consistency.
3. **Hosting & CI/CD:** Deployed the containerized service to Hugging Face Spaces, with fully automated build and deployment workflows managed via GitHub Actions.

### Stack
`Python`, `OpenCV`, `FastAPI`, `Docker`, `GitHub Actions`