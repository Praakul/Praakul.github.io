---
title: "Cylindrical Suturing Evaluation System"
date: 2026-06-15
draft: false
projectType: "Work"
tags:
- Agentic AI
- PyTorch
- FastAPI
- PyQt5
description: "Res2Net ensemble of 7 specialized models for geometric suturing metrics (slack, angulation, edge alignment) paired with an LLM agentic cloud pipeline that converts scores into clinical feedback."
---
## TL;DR
A robust, multi-modal system designed to evaluate the quality of medical suturing (both "Flat" and "Cylindrical") performed by practitioners. It combines an ensemble of localized edge PyTorch models for instant metric calculations with a cloud-based Agentic AI framework for comprehensive qualitative evaluation.

## The Complexity of Multi-Variable Assessment
Suturing evaluation requires assessing multiple independent geometric variables simultaneously (e.g., Inter-suture Distance, Slack, Position, Angulation, Edge Alignment). A single neural network struggles to reliably predict all these continuous variables at once. Furthermore, raw numbers (e.g., "Slack = 0.4") are unhelpful to a medical trainee; they need qualitative, actionable feedback explaining how to fix their technique.

## Bridging Quantitative Metrics and Clinical Feedback
To achieve high accuracy on the metrics, the problem had to be broken down. We chose an ensemble architecture where specialized models handle specific visual tasks. However, generating human-readable medical feedback from 7 independent metric floats requires semantic reasoning. Therefore, the architecture was split: the edge device (client) runs the deterministic computer vision models for speed and privacy, while the server runs an LLM-based Agentic AI pipeline to synthesize those metrics into a comprehensive clinical report.

## Edge Ensembles and Cloud Synthesis

The system is decoupled into a PyQt5 Client (Edge) and a FastAPI Server (Cloud).

### The Edge Client (PyTorch Ensemble)
The client interface handles the live webcam feed via OpenCV. When the user requests an evaluation, the image is passed through an ensemble of **seven different Res2Net-based checkpoints**. Instead of one massive model, each checkpoint specializes in a specific feature (e.g., one specifically trained to measure tension/slack, another for edge alignment). This separation of concerns drastically improves metric accuracy.

### The Cloud Server (Agentic AI)
The client transmits the calculated metric arrays to the FastAPI backend. 
- **LLM Synthesis**: An Agentic AI pipeline ingest the raw scores and cross-references them against medical suturing standards. It uses Large Language Models to convert quantitative data (e.g., erratic inter-suture distances) into qualitative, actionable feedback (e.g., "Your needle bite depth is inconsistent, causing edge misalignment").
- **Report Generation**: The server formats this synthesized feedback into Markdown, utilizes a custom `PDFReportExporter` to generate the final document, and integrates with an SMTP service to asynchronously email the evaluation to the practitioner.
