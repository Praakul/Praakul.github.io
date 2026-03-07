---
title: "Medical Model Calibration"
date: 2026-01-10
draft: false
tags: ["Deep Learning", "Medical Imaging", "Model Calibration", "Trustworthy AI"]
badges: ["PyTorch", "MDCA", "Focal Loss", "Scikit-learn"]
links:
  - icon: fab fa-github
    url: https://github.com/praakul/medicalModelCalibration
---

In high-stakes medical imaging, high accuracy alone is dangerous. A model predicting "99% confidence" must actually be correct 99% of the time. Most deep learning models are **overconfident** — they concentrate probability mass on their predictions regardless of actual correctness. This project implements a comprehensive pipeline to evaluate and fix this problem, comparing two fundamentally different approaches across **180 experiments**.

---

## The Core Question

> _Is it better to train a model that's calibrated from the start, or fix miscalibration after training?_

**Answer: Train-time regularization wins decisively.** The combination of **Focal Loss + MDCA** achieved 97.8% accuracy with minimal calibration error — no post-processing needed.

---

## Train-Time Approach: MDCA Loss

The key innovation is the **MDCA (Multi-class Difference in Confidence and Accuracy)** auxiliary loss function. For each class `c`, MDCA computes:

```
loss += |avg_confidence(c) - avg_frequency(c)|
```

This penalizes the gap between a model's average predicted probability for class `c` and the actual proportion of class `c` in the batch. The total loss is normalized by the number of classes.

**Implementation:** The `CombinedLoss` class combines a primary loss (CrossEntropy or Focal Loss) with the MDCA regularizer weighted by β:

```
total_loss = primary_loss(logits, targets) + β × MDCA(logits, targets)
```

Four loss configurations were tested:
- `cross_entropy` — Standard baseline
- `focal_loss` — Down-weights easy examples using `(1-pt)^γ` modulation (γ=2.0)
- `NLL+MDCA` — CrossEntropy + MDCA (β=5.0)
- `FL+MDCA` — Focal Loss + MDCA (β=5.0) ← **State-of-the-art result**

---

## Post-Hoc Approach: Temperature & Dirichlet Scaling

Two post-hoc methods were implemented for comparison:

### Temperature Scaling
A single learnable parameter `T` (initialized at 1.5) divides the logits before softmax: `scaled_logits = logits / T`. The temperature is optimized on the validation set using **LBFGS** (max 100 iterations) to minimize cross-entropy loss. The parameter is clamped to `min=1e-4` to prevent division by zero.

### Dirichlet Scaling (Matrix Scaling)
A learned `nn.Linear(num_classes, num_classes)` layer transforms the logits. This is more expressive than temperature scaling — it can learn per-class scaling relationships. Includes an L2 regularizer on the bias term weighted by µ to prevent overfitting on small validation sets. Also optimized with LBFGS.

**Critical finding:** Dirichlet scaling introduces **massive variance** on small medical validation sets. The extra parameters overfit to the validation distribution, often *degrading* the model's reliability compared to no calibration at all.

---

## Experiment Design

**180 total experiments** = 5 datasets × 3 architectures × 4 loss functions × 3 configurations (no calibration, temperature, dirichlet)

### Architectures
- ResNet-18, ResNet-34, ResNet-50 (all with configurable dropout)

### Datasets

| Dataset | Classes | Size | Labels |
|---------|---------|------|--------|
| Breast Ultrasound | 3 | 780 | Benign, Malignant, Normal |
| Surgical Skills | 3 | 1,216 | Average, Good, Poor |
| Brain Tumour | 4 | 7,023 | Glioma, Meningioma, No Tumor, Pituitary |
| Chest X-Ray | 2 | 5,840 | Normal, Pneumonia |
| COVID-19 | 3 | 12,098 | COVID, Normal, Viral Pneumonia |

The data loader automatically discovers the train/test split from directory structure and creates a validation set.

### Metrics Suite
Every experiment tracks:
- **Performance:** Accuracy, Weighted F1-Score, Weighted AUC (One-vs-Rest)
- **Calibration:** ECE (Expected Calibration Error), MCE (Maximum Calibration Error), ACE (Adaptive Calibration Error using equal-count bins)
- **Visualization:** Reliability diagrams (confidence vs. accuracy per bin) saved for every best model

### Training Pipeline
- **Optimizer:** Adam with configurable learning rate and weight decay
- **Scheduler:** MultiStepLR with configurable milestones and gamma
- **Automation:** A shell script (`autoTrainer.sh`) loops through all dataset-loss-model combinations, running `train.py` for each. A separate `autoCalibrator.py` applies post-hoc methods to every trained checkpoint.
- **Results compilation:** `compileResults.py` parses all experiment logs into a CSV for analysis. `plotresults.py` generates publication-ready comparison plots (calibration landscapes, method comparison boxplots, loss impact analysis).

---

## Stack
`Python` · `PyTorch` · `Torchvision` · `Scikit-learn` · `Pandas` · `Seaborn` · `Matplotlib` · `NumPy`