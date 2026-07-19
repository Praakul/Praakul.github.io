---
title: Medical Model Calibration
date: 2026-01-10
draft: false
projectType: "Projects"
tags:
- Deep Learning
- Medical Imaging
- Model Calibration
- Trustworthy AI
badges:
- PyTorch
- MDCA
- Focal Loss
- Scikit-learn
description: "180 experiments across 5 medical datasets proving that Focal Loss + MDCA train-time regularization beats post-hoc Temperature Scaling for neural network calibration."
repoURL: https://github.com/Praakul/medicalModelCalibration
---

## TL;DR
In medical imaging, raw accuracy is not enough—it can be fatal. If a diagnostic model claims it is "99% confident" about a benign tumor, but is mathematically only correct 60% of the time in those scenarios, a doctor might skip a life-saving biopsy. This project involved 180 rigorous experiments across 5 datasets to evaluate and solve the systemic overconfidence of deep neural networks using both train-time regularizers and post-hoc scaling techniques.

## The Danger of Overconfidence
Neural networks output raw, unbounded numbers called **logits**. The Softmax function converts these logits into a 100% probability distribution. However, modern networks are notoriously "overconfident," mathematically inflating the probability of their winning class (e.g., predicting a disease with 99.9% confidence even on blurry scans). Calibration is the process of aligning a model's stated confidence with its actual accuracy. A perfectly calibrated model that says it is 80% confident should be exactly right 8 out of 10 times.

## The Focal Loss Paradox
The core motivation is functional safety. In healthcare, worst-case scenarios matter more than averages. Standard metrics like Accuracy hide terrible outliers, especially in imbalanced datasets (e.g., rare malignant tumors). To truly evaluate safety, we must use metrics like **Expected Calibration Error (ECE)** (average honesty) and **Maximum Calibration Error (MCE)** (the most dangerous lie the model tells). 

Furthermore, popular techniques designed to solve class imbalance—specifically **Focal Loss**—are mathematically guaranteed to destroy calibration. Focal Loss drops the penalty for "easy" examples, forcing the network to artificially inflate its confidence just to appease the loss function on "hard" examples. This paradox requires a robust architectural fix.

## Benchmarking Train-Time vs. Post-Hoc Fixes

To definitively answer whether it is better to train a model to be calibrated out-of-the-box or to patch its overconfidence post-training, I implemented and benchmarked multiple techniques:

### Post-Hoc Scaling (The Patch)
1. **Temperature Scaling**: Uses a single scalar ($T$) to divide logits before Softmax. This acts as a global "confidence dial" that squishes probabilities closer together without changing the winning class (preserving accuracy).
2. **Dirichlet Scaling**: Uses a fully learned linear matrix transformation ($W \cdot \ln(p) + b$) to adjust confidence per-class based on historical biases. 
*Implementation Result*: Dirichlet scaling proved erratic on small medical datasets, heavily overfitting to validation sets and often making reliability worse.

### Train-Time Regularization (The Solution)
To counteract the calibration destruction caused by Focal Loss, I implemented the **MDCA (Multi-class Difference in Confidence and Accuracy)** auxiliary loss function.

```python
total_loss = Focal_Loss(logits, targets) + β × MDCA(logits, targets)
```

**The Engineering Reality:**
- **Focal Loss** forces the network to focus on finding rare, hard diseases (keeping Weighted F1 and AUC high).
- **MDCA Regularizer** acts as a mathematical leash, penalizing the model if its average confidence for a class drifts away from how often it is actually correct.

By pairing Focal Loss with MDCA (`FL+MDCA`), the model achieved 97.8% accuracy while dropping the Expected Calibration Error (ECE) by a massive -0.0318 compared to raw Focal Loss. This proved that train-time regularization decisively beats post-hoc hacks for medical deployments.