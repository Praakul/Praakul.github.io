---
title: "Medical Model Calibration in Deep Neural Networks"
date: 2026-01-10
draft: false
tags: ["Deep Learning", "Clinical Imaging", "Model Calibration"]
---

### The Objective
High-risk clinical decision-making requires models that are accurate, but more importantly, properly calibrated. I conducted a study focused on reducing overconfidence in deep neural networks used for clinical imaging.

### Technical Approach
* **Core Challenge:** Standard DNNs often output overconfident probability estimates, which is dangerous in a medical context.
* **Implementation:** I integrated auxiliary loss functions directly into the training pipeline to improve the baseline probability estimates.
* **Benchmarking:** The custom loss function approach was benchmarked against standard post-hoc methods, specifically temperature scaling, to validate the improvement in expected calibration error (ECE).

### Stack
`Python`, `PyTorch`, `Scikit-learn`