---
title: "Medical Model Calibration"
date: 2026-01-10
draft: false
tags: ["Deep Learning", "Medical Imaging", "Model Calibration", "Trustworthy AI"]
badges: ["PyTorch", "MDCA", "Focal Loss", "Scikit-learn"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul/medicalModelCalibration
---

In the world of medical imaging, having high accuracy just isn't enough—it can actually be dangerous. If a model says it's "99% confident" about a diagnosis, it needs to be right 99% of the time. The catch is that most deep learning models are wildly **overconfident**, throwing high probabilities at predictions even when they're wrong. I built a comprehensive pipeline to tackle this exact problem, running **180 experiments** to compare two completely different ways of fixing neural network calibration.

---

### The Core Question

> _Is it better to train a model to be calibrated right out of the box, or to patch its overconfidence after training?_

**The Answer: Train-time regularization wins decisively.** Using a combo of **Focal Loss and MDCA** hit 97.8% accuracy with barely any calibration error, completely eliminating the need for post-processing hacks.

---

### Fixing It During Training: MDCA Loss

My favorite approach relies on the **MDCA (Multi-class Difference in Confidence and Accuracy)** auxiliary loss function. For every class `c`, it checks:

```text
loss += |avg_confidence(c) - avg_frequency(c)|
```

Basically, it heavily penalizes the model if its average confidence for a class doesn't match how often that class actually appears. 

**How I implemented it:** I wrote a `CombinedLoss` class that pairs a primary loss (like standard CrossEntropy or Focal Loss) with the MDCA regularizer.

```text
total_loss = primary_loss(logits, targets) + β × MDCA(logits, targets)
```

I tested four different configurations:
- `cross_entropy` — The basic baseline.
- `focal_loss` — Drops the weight of "easy" examples.
- `NLL+MDCA` — CrossEntropy plus MDCA.
- `FL+MDCA` — Focal Loss plus MDCA (**This achieved the state-of-the-art result**).

---

### Fixing It After Training: Post-Hoc Scaling

For comparison, I also implemented two popular methods for fixing calibration *after* a model is already trained:

#### Temperature Scaling
This is the classic trick: divide the model's logits by a single, learnable parameter `T` (starting at 1.5) right before the softmax layer. I used **LBFGS** to optimize `T` on the validation set so it minimizes the cross-entropy loss.

#### Dirichlet Scaling (Matrix Scaling)
This steps it up by using a fully learned linear layer to transform the logits, letting it discover complex scaling relationships per class. To stop it from dramatically overfitting on small validation sets, I added an L2 regularizer.

**What I found:** Dirichlet scaling behaves erratically on small medical datasets. Because it has extra parameters, it ends up overfitting strictly to the validation data. In many cases, it made the model's reliability *worse* than doing nothing at all!

---

### How the Experiments Were Designed

I ran **180 total experiments** to be absolutely sure of the results. This covered 5 datasets, 3 model architectures, 4 loss functions, and 3 post-hoc setups.

#### Architectures
- ResNet-18, ResNet-34, and ResNet-50 (all tweaked with configurable dropout).

#### Datasets

| Dataset | Classes | Size | Labels |
|---------|---------|------|--------|
| Breast Ultrasound | 3 | 780 | Benign, Malignant, Normal |
| Surgical Skills | 3 | 1,216 | Average, Good, Poor |
| Brain Tumour | 4 | 7,023 | Glioma, Meningioma, No Tumor, Pituitary |
| Chest X-Ray | 2 | 5,840 | Normal, Pneumonia |
| COVID-19 | 3 | 12,098 | COVID, Normal, Viral Pneumonia |

I also wrote a neat data loader that automatically figures out the train/test split from the directory structure and handles validation set creation.

#### Tracking the Metrics
Every single run tracked:
- **Performance:** Accuracy, Weighted F1-Score, Weighted AUC.
- **Calibration:** ECE (Expected Calibration Error), MCE, and ACE.
- **Visualization:** I set it up to auto-generate reliability diagrams for the best models.

#### The Training Pipeline
Everything was heavily automated:
- **Optimizer & Scheduler:** Adam optimizer paired with a MultiStepLR scheduler.
- **Automation script:** A bash script (`autoTrainer.sh`) iterated through every possible dataset-loss-model combo, and a separate `autoCalibrator.py` script reliably applied the post-hoc fixes to all the checkpoints.
- **Analytics:** At the end, a `compileResults.py` script parsed hundreds of logs right into a clean CSV, and `plotresults.py` spat out publication-ready charts comparing calibration methods.

---

### Tech Stack
`Python` · `PyTorch` · `Torchvision` · `Scikit-learn` · `Pandas` · `Seaborn` · `Matplotlib` · `NumPy`