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
    url: https://huggingface.co/spaces/PrajwalKulkarni/vehicleTracking
---

I built a completely from-scratch implementation of the **SORT (Simple Online and Realtime Tracking)** algorithm to track multiple vehicles in traffic videos. Instead of just importing a library, I really wanted to understand the math behind it, so I wrote every core component—like the Kalman Filter, the Hungarian assignment solver, and the track lifecycle manager—from the ground up. I then packaged it as a production-ready FastAPI service and deployed it on Hugging Face Spaces.

---

### Understanding the Kalman Filter

At the heart of the project is a **6-dimensional Kalman Filter** that keeps track of each vehicle using a constant-velocity motion model.

#### The State Vector
```text
x = [cx, cy, w, h, vx, vy]
```
Here, `cx` and `cy` represent the center of the bounding box, `w` and `h` are its width and height, and `vx` and `vy` show how fast the center is moving.

#### Predicting the Next State
To guess where the vehicle will be in the next frame, the transition matrix `F` assumes the velocity stays constant:
```text
cx' = cx + vx·dt    (position updates based on velocity)
cy' = cy + vy·dt
w'  = w              (size stays the same between frames)
h'  = h
vx' = vx             (velocity remains constant)
vy' = vy
```
The uncertainty (state covariance) also updates like this: `P' = F·P·Fᵀ + Q`.

#### Updating with Measurements
When YOLO detects a vehicle, we only get its position and size (`[cx, cy, w, h]`). We have to infer its velocity using the filter's internal state. 

Here's how the Kalman equations update our predictions:
1. **Innovation:** `y = measurement - H·state` (Figuring out how wrong our prediction was)
2. **Innovation covariance:** `S = H·P·Hᵀ + R`
3. **Kalman Gain:** `K = P·Hᵀ·S⁻¹` (Deciding whether to trust the new measurement or our prediction more)
4. **State update:** `state = state + K·y`
5. **Covariance update:** `P = (I - K·H)·P`

I tuned the noise values specifically for traffic videos. Since vehicles usually follow predictable paths but can change speed, I set the process noise `Q` with `q_pos=0.1` and `q_vel=1.0`. For the measurement noise `R`, a value of `5.0·I` worked well to balance out YOLOv8's detection quirks.

---

### How the SORT Pipeline Works

For every single frame of the video, things go through four main steps:

#### 1. Spotting the Vehicles (Detection)
A pre-trained **YOLOv8n** model scans the frame and pulls out detections. I filtered these to only care about COCO vehicle classes: cars, buses, and trucks. Anything with low confidence gets tossed out.

#### 2. Predicting Where They're Going
Every active track uses its own Kalman Filter to guess where it expects to see the vehicle in the current frame. The `time_since_update` counter also ticks up for each track.

#### 3. Matching Detections to Tracks (Hungarian Algorithm)
This is where the magic happens. We have N predicted positions and M new detections from YOLO, and we need to pair them up perfectly. 
- I calculate a **cost matrix** measuring the mismatch between predictions and detections (`1 - IoU(predicted, detected)`).
- Then, the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) jumps in to solve this as a minimum-cost assignment puzzle.
- Any matches with an IoU of less than 0.3 are thrown out to prevent weird associations.

#### 4. Managing the Tracks
- **Matched detections:** We update their Kalman Filter and hit reset on the `time_since_update`.
- **Unmatched detections:** These get a brand new `Track` and a fresh Kalman Filter with a new ID.
- **Unmatched tracks:** If a vehicle is missing for a few frames (maybe passing behind a sign), I keep it alive for up to 5 frames before finally deleting it.

In the end, this gives us a solid list of active vehicles, each sporting a stable ID and a smooth, tracked trajectory.

---

### Taking It to Production

I didn't just want this to live in a Jupyter notebook, so I containerized the whole pipeline and shipped it as an API:

- **FastAPI** manages the async file uploads, runs the processing, and serves up the final video.
- **Docker** sets everything up cleanly with a multi-stage build.
- **Hugging Face Spaces** hosts it publicly using their Docker SDK.
- **GitHub Actions** handles the simple CI/CD pipeline, automatically redeploying whenever I push to `main`.
- **Interactive docs** at `/docs` let you immediately test it out with your own videos.

#### Try it out!
**Via API:**
```bash
curl -X POST "https://prajwalkulkarni-vehicletracking.hf.space/track_video/" \
     -F "file=@your_video.mp4" -o "tracked_output.mp4"
```

Or just head over to the [live interactive docs](https://prajwalkulkarni-vehicletracking.hf.space/docs) to give it a spin.

---

### Tech Stack
`Python` · `NumPy` · `SciPy` · `OpenCV` · `Ultralytics (YOLOv8)` · `FastAPI` · `Docker` · `GitHub Actions`