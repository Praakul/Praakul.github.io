---
title: Surgical Skill Assessment System
date: 2025-08-15
draft: false
projectType: "Projects"
tags:
- FastAPI
- PyQt5
- Client-Server
- Video Processing
- Async
badges:
- FastAPI
- PyQt5
- AsyncIO
- SMTP
description: "Fault-tolerant distributed system for AIIMS Delhi — PyQt5 edge client with network-resilient uploads and a FastAPI async job queue using ThreadPoolExecutors for concurrent video scoring."
repoURL: https://github.com/Praakul/surgicalSkillAnalysisSystem
---

## TL;DR
I engineered a production-ready, highly fault-tolerant distributed system for the Neuro-Engineering Lab at AIIMS Delhi. The platform allows surgical trainees to record, submit, and automatically receive objective, computer-vision-based evaluations of their surgical exercises. It bridges the gap between unreliable hospital network infrastructure and heavy asynchronous ML processing.

## Scaling Objective Surgical Feedback
Surgical training requires objective, repeatable evaluation of procedural skills via video, but traditional observation is subjective, expensive, and unscalable. The technical challenge is capturing massive high-resolution video streams from multiple edge clients (trainees) and reliably transmitting them to a central server for intensive AI processing—all while ensuring no data is lost during the frequent network drops common in medical facilities.

## Surviving Hospital Network Drops
A standard monolithic web application fails under these conditions. If a heavy PyTorch inference job runs on the main server thread, it blocks incoming HTTP requests, crashing the system for all other users. We required a decoupled client-server architecture. The GUI had to be foolproof (designed for non-technical surgeons), and the backend had to elegantly separate network I/O from CPU-bound Machine Learning tasks, using asynchronous event loops and ThreadPoolExecutors.

## Asynchronous I/O and Queue Architecture

The system consists of a PyQt5 edge client and an asynchronous FastAPI central server.

### The PyQt5 Edge Client
Built to be bulletproof for clinical environments:
- **Asynchronous Video Pipeline**: Uses a `QThread` and `QTimer` firing at ~30ms to pull frames via OpenCV without blocking the main UI event loop. 
- **Network-Resilient Uploads**: Implements a dedicated `VideoSender` thread using `requests-toolbelt` for byte-level multipart streaming. If the hospital Wi-Fi drops, it utilizes an exponential back-off algorithm (up to 3 retries) and allows mid-stream pausing/resumption using a background socket-probing daemon (`NetworkMonitor`).

### The FastAPI Async Server
The backend strictly separates I/O from compute to maintain extreme concurrency:
- **The Event-Driven Job Queue**: A custom `JobQueue` class manages the lifecycle. Instead of expensive CPU polling, it uses an `asyncio.Event` to instantly wake up the processor the microsecond a new video payload arrives.
- **Thread Pool Execution**: Heavy Computer Vision models are intentionally dropped into a `ThreadPoolExecutor` via `loop.run_in_executor()`. This keeps the FastAPI `uvicorn` event loop completely unblocked to handle new REST requests (like `/status` polling). Thread-safe state mutation is guaranteed using `asyncio.Lock` and `asyncio.run_coroutine_threadsafe`.

### The Telemetry & Notification Engine
Once a video is processed, the system connects to an SMTP relay to email the trainee their exact score breakdown. If the network drops the exact second a job finishes, the system transitions the job to a `PENDING_EMAIL` state. An independent `periodic_maintenance` task sweeps the queue, retrying deliveries to guarantee no trainee ever loses their evaluation.
