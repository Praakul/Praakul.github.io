---
title: "My Research Internship at AIIMS New Delhi"
date: 2025-09-30
draft: false
tags: ["AIIMS", "Internship", "Neurosurgery", "AI", "Research"]
---

In May 2025, I walked through the gates of the **All India Institute of Medical Sciences, New Delhi** — India's premier medical institution — as a research intern at the **Neuro-Engineering Lab**, Department of Neurosurgery. What followed was a five-month journey that taught me more about building real AI systems than any course ever could.

---

## The Setting

The **Neurosurgery Education and Training School (NETS)** at AIIMS is where neurosurgeons train. They progress through a carefully structured curriculum: flat suturing on synthetic sheets → cylindrical tube suturing → nerve and vessel suturing on rat models → craniotomy drilling on synthetic material, sheep heads, and scapula. Each stage uses increasingly thinner threads and higher precision requirements across 10+ iterations.

NETS has been recording these training sessions since 2015. Hundreds of hours of video. Thousands of training samples. The lab's mission: **can AI reliably evaluate these procedures and provide trainees with immediate, quantitative feedback?**

I worked on three independent projects targeting different stages of this training pipeline, each presenting fundamentally different engineering challenges.

---

## Project 1: Craniotomy Evaluation — The Deployment Challenge

### The Problem
Craniotomy — drilling through the skull to access the brain — is a prerequisite for nearly every neurosurgery. A deep learning model had already been developed by the lab ([published in Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/abs/pii/S0010482525310017)) to score drilling quality from microscopic images. The model existed as research code. My job was to turn it into something surgeons could actually use.

### The Technical Reality
The drilling lab at AIIMS has unreliable internet. Surgeons need something that works offline, instantly, on any available laptop. A web app was never an option. It had to be a **standalone desktop application**.

I built it in **PyQt5** — chosen specifically because it integrates natively with Python's scientific stack (OpenCV, PyTorch) and provides a signal-slot architecture suitable for concurrent video processing without blocking the UI.

### Implementation Details

**The GUI** has 63,000+ lines of code across two tabs:

**Photo Mode:**
- Opens the system file manager for image selection
- The critical step: **homographic transformation**. After selecting an image, a popup window asks the user to click 4 corner points on the drilling sample. These are fed to `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to produce a rectified, cropped image of just the drilling area. Without this, the model would waste capacity on the background — the microscope housing, the bench, the surgeon's hands.
- The transformed image is passed through the **Res2Net** backbone (a multi-scale feature extraction architecture that captures drilling patterns at different granularities) for inference
- Score appears instantly in the UI

**Camera Mode:**
- A `QThread`-based video pipeline that creates a new thread every time a camera is activated, preventing frame processing from blocking the UI event loop
- Camera source selection dropdown (supporting multiple connected cameras — useful because the lab has both microscope cameras and webcams)
- Capture, record (with file save dialog), and evaluate controls

**Grad-CAM — The Feature Surgeons Actually Wanted:**
Surgeons don't trust a number. They need to *see* what the model considers good and bad. I integrated **Gradient-weighted Class Activation Mapping** using the final convolutional layer:
1. Forward pass through the network, recording feature maps
2. Backpropagate the predicted class score
3. Global average pool the gradients to get per-channel importance weights
4. Weight the feature maps, ReLU, and resize to the input image
5. Overlay as a heatmap: green = high-quality drilling, red = areas needing improvement

This single feature did more for surgeon trust than any accuracy metric.

**Report Generation:**
- A structured HTML template (with CSS styling) is populated with the evaluation image, Grad-CAM overlay, scores, and trainee metadata
- Rendered to PDF and automatically emailed to the trainee's email address using SMTP
- Both "Satisfied" and "Unsatisfied" flows save the data — the unsatisfied path additionally captures the surgeon's own score, building a correction dataset for model improvement

The application was **deployed across the AIIMS drilling lab** by the end of the internship.

---

## Project 2: Cylindrical Suturing — The Agentic AI Experiment

### The Problem
Cylindrical suturing is the next training stage after flat sheets. Trainees suture on both sides of a synthetic tube (they flip it halfway through), then progress to actual nerves and vessels on animal models. Each sample needs evaluation: stitch placement, angle consistency, tension uniformity, knot quality.

The catch: **no annotated dataset existed.** Years of training videos, but no labeled images.

### Data Collection — Going Through 10 Years of Video
I manually watched recorded training videos on the NETS server — from 2015 to May 2025. For each suturing exercise, I captured a screenshot after the final stitch. Since doctors suture both sides of the cylinder, each exercise yields 2 images.

Final dataset:
- **394 images** — synthetic tube normal joining
- **26 images** — synthetic tube bypass suturing
- **94 images** — actual nerve and vessel sutures
- **Total: 514 images** across dozens of trainees and residents

With 514 images and no labels, supervised learning was off the table. We needed a different approach.

### The Multi-Agent Architecture
We built a **Manager-Agent system** using multimodal large language models:

**Agent Selection:** We chose two open-source multimodal LLMs from Meta's LLaMA family. LLaMA Maverick (the larger model, better multimodal benchmarks) serves as both the Manager and one Expert. LLaMA Scout (smaller, faster) acts as the second Expert. API calls go through **Groq** (for GPU access without local hardware costs), and **Agno** handles multimodal image inputs.

**Evaluation Flow:**
1. Test image is submitted to both Expert Agents independently
2. Each Expert evaluates the suture across predefined criteria (position, angulation, tension, knot quality, symmetry)
3. The Manager Agent receives both evaluations, resolves disagreements, and produces a final score with a descriptive report explaining the reasoning

### Visual RAG — Giving Agents Reference Context
Without fine-tuning, general-purpose LLMs have no concept of what a "good" suture looks like. We addressed this with **Visual RAG**:

1. 50 reference suture images (graded by expert neurosurgeons) are encoded into vector embeddings using **CLIP's vision transformer**
2. These embeddings form a searchable knowledge base
3. For each new test image, CLIP encodes it and retrieves the 3 most visually similar references
4. These references (with their expert scores) are provided to both agents as in-context examples
5. Agents are instructed to calibrate their evaluation relative to these references — not copy scores, but use them as anchors

### The Results — An Honest Account

| Metric | Value |
|--------|-------|
| RMSE vs. expert scores | 2.2 / 10 |
| ±1 Accuracy | 52% |
| RMSE of expert's subscores vs. own total | 2.58 / 10 |

The 52% ±1 accuracy is underwhelming. The 2.2 RMSE means the model is, on average, off by more than 2 points on a 10-point scale.

But there's an interesting nuance: the variance between an expert's own subscores and their final score (2.58 RMSE) is *higher* than the model-expert disagreement (2.2 RMSE). Human scoring of suture quality is inherently subjective.

The honest takeaway: small multimodal LLMs, even with RAG and multi-agent orchestration, are not sufficient for this level of domain specificity. The architecture is sound. The model size is the bottleneck. This informed the lab's decision to pursue supervised approaches with domain-specific fine-tuning going forward.

---

## Project 3: Endoscopy Eye Tracking — The Hardware Project

### The Problem
A parallel research question: do expert neurosurgeons develop distinct gaze patterns during endoscopic procedures? Can we measure these patterns and use them as objective skill markers?

This requires synchronized, high-framerate recording of a surgeon's eyes and hand movements during endoscopic training. No off-the-shelf solution existed for the lab's specific setup.

### What I Built
A three-camera synchronized recording system:

**Hardware:** Two **ESP32-S3** microcontroller boards — small enough to mount near a surgeon's eyes during endoscopic exercises — and one USB webcam for workspace capture.

**Firmware:** Custom C code, developed in Arduino IDE, for ESP32-S3 configuration:
- Frame capture at the required FPS
- USB video streaming (wired, not WiFi — I tested both, and WiFi produced ~15% frame drops and 200+ ms latency, unacceptable for gaze analysis where timing precision matters)
- Camera register configuration for resolution and exposure settings

**Software:** A Python GUI with OpenCV managing:
- Three simultaneous video inputs
- Synchronized recording start/stop across all feeds
- Individual stream display and controls

The system was **deployed in the AIIMS endoscopy lab**. My internship ended before the full analysis phase, but the recording infrastructure I built is being used for ongoing data collection.

---

## What I Took Away

**Deployment is a different discipline.** The craniotomy project taught me that 80% of the engineering effort in real AI systems isn't the model — it's the homographic transformation dialog, the PDF template, the email integration, the camera thread management, the feedback loop. The model was already built. Making it usable by surgeons who have zero interest in debugging Python was the actual challenge.

**Research honesty matters.** Reporting 52% accuracy for the suturing project was uncomfortable. But that honest result changed the lab's research direction — away from general-purpose LLMs and toward supervised fine-tuning with domain-specific data. Inflating results would have wasted months of future work.

**Domain context beats model complexity.** The Grad-CAM feature wasn't the most technically complex thing I built, but it was the feature that made surgeons actually trust and use the system. Understanding what surgeons need (visual explanations, not numbers was a critical need) required sitting in the lab, watching them use prototypes, and iterating.

I am grateful to **Prof. Ashish Suri**, **Dr. Ramandeep Singh**, **Dr. Rohan Dhanakshirur** (my supervisor), **Mr. Anuj Saini**, and the entire Neuro-Engineering Lab team at AIIMS for this extraordinary opportunity.
