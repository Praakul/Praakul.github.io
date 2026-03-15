---
title: "Surgical Skill Assessment System"
date: 2025-08-15
draft: false
tags: ["FastAPI", "PyQt5", "Client-Server", "Video Processing", "Async"]
badges: ["FastAPI", "PyQt5", "AsyncIO", "SMTP"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul/surgicalSkillAnalysisSystem
---

I built a production-ready, multi-client, multi-server system designed to record, process, and evaluate surgical training procedures. I developed this specifically for the Neuro-Engineering Lab at AIIMS Delhi, giving trainees a seamless way to record their surgical exercises, submit them to a centralized server, and automatically receive their scored feedback via email.

---

### The Architecture

```text
┌─────────────────────┐         ┌─────────────────────┐         ┌────────────────┐
│ Client (PyQt5)      │         │ Server (FastAPI)     │         │ Email Service  │
│ - Video recording   │────────▶│ - REST API           │────────▶│ - SMTP relay   │
│ - User metadata     │◀────────│ - Async job queue    │         │ - Score report │
│ - Upload w/ retry   │         │ - Thread pool        │         └────────────────┘
└─────────────────────┘         └─────────────────────┘
```

---

### The Client: A PyQt5 Desktop App

I designed the desktop GUI to be foolproof, considering it was built for surgeons and trainees who just want things to work without dealing with technical hiccups.

#### Video Recording Pipeline
- There's a live webcam preview driven by OpenCV in a background `QThread` to keep the UI snappy.
- It includes super clear **Start / Pause / Resume / Stop** controls and visual indicators so they always know if they're recording.
- Hitting "Stop & Save" automatically pauses the stream, suggests a clean timestamped filename, and safely stores the file.

#### User Information Form
- I used a `QSplitter` to give the video preview 70% of the screen and the metadata form 30%.
- It collects standard info like their name, program, and iteration number, and actually rigorously validates inputs before letting them upload anything.

#### Network-Resilient Uploads
Hospitals don't always have the best Wi-Fi, so the upload had to be bulletproof. It runs in a dedicated thread using a robust **signal-slot architecture**.
- It features live progress bars, clear success/error dialogs, and a live network status indicator checking connectivity.
- A user can cancel an upload or easily hit "Retry Connection" to resume a paused transfer exactly where it left off.

---

### The Server: FastAPI and Async Queues

#### REST API Endpoints
| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/submit` | POST | Accepts the video and metadata payload |
| `/status/{id}` | GET | Checks if the job is queued, processing, done, or failed |
| `/queue-status` | GET | Shows the queue length and accurate ETA |
| `/submission/{id}` | DELETE | Safely cancels a queued job |
| `/health` | GET | Simple server ping |

#### The Core Engine: Job Queue
The `JobQueue` class is the brains of the operation. It elegantly handles concurrent video processing without breaking a sweat:
1. **Thread Pool Executor:** I set it up to process multiple videos simultaneously using Python's `ThreadPoolExecutor`. Crucially, this runs purely in the background via `loop.run_in_executor()` so the main FastAPI event loop never gets blocked.
2. **Safe Concurrency:** An `asyncio.Lock` ensures the queue doesn't trip over itself when multiple clients spam submissions at once.
3. **Event-Driven:** Instead of blindly polling, an `asyncio.Event` immediately wakes up the processor the second a new video lands.
4. **Network Awareness:** Before tackling a heavy video, the queue makes sure the internet is actually up so it can eventually email the results. If the network is down, jobs are safely paused, not dropped.

#### Email Service
As soon as a video is scored, my SMTP integration fires off an automated email to the trainee with their results. If the internet drops the second a job finishes, it just waits in a `PENDING_EMAIL` state and tries again later.

---

### Handling the Edge Cases

I poured a lot of thought into what happens when things go wrong:
- **Network drops:** The client will automatically retry uploads, and the server pauses emails instead of losing them into the void.
- **Traffic spikes:** Thread-safe locking and configurable executors ensure the server doesn't crash if an entire class uploads at once.
- **Messy exits:** The client safely releases the camera and cleans up temp files even if a user rudely forces the window closed mid-recording.

---

### Tech Stack
`Python` · `FastAPI` · `PyQt5` · `OpenCV` · `AsyncIO` · `ThreadPoolExecutor` · `SMTP` · `Uvicorn`
