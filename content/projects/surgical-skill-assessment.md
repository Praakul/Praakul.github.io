---
title: "Surgical Skill Assessment System"
date: 2025-08-15
draft: false
tags: ["FastAPI", "PyQt5", "Client-Server", "Video Processing", "Async"]
badges: ["FastAPI", "PyQt5", "AsyncIO", "SMTP"]
links:
  - icon: fab fa-github
    url: https://github.com/Praakul
---

A production multi-client, multi-server system for recording, submitting, processing, and evaluating surgical training procedures. Built for the Neuro-Engineering Lab at AIIMS Delhi — where multiple trainees submit recorded videos of their surgical exercises for automated scoring and email-based feedback.

---

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐         ┌────────────────┐
│ Client (PyQt5)      │         │ Server (FastAPI)     │         │ Email Service  │
│ - Video recording   │────────▶│ - REST API           │────────▶│ - SMTP relay   │
│ - User metadata     │◀────────│ - Async job queue    │         │ - Score report │
│ - Upload w/ retry   │         │ - Thread pool        │         └────────────────┘
└─────────────────────┘         └─────────────────────┘
```

---

## The Client — PyQt5 Desktop Application

A full-featured desktop GUI designed for surgical trainees with zero technical background:

### Video Recording Pipeline
- Live webcam preview via the `VideoWidget` component (based on OpenCV capture in a background `QThread`)
- **Start / Pause / Resume / Stop** controls with visual recording indicator (red = recording, orange = paused, green = saved)
- On "Stop & Save," the application pauses recording, opens a file dialog for the user to choose a save location (with auto-generated timestamp filename `surgical_recording_YYYYMMDD_HHMMSS.mp4`), then saves and releases

### User Information Form
- **70/30 split layout** using `QSplitter` — video preview takes 70% of the window, the form panel takes 30%
- Fields: Name (validated — no numbers/special characters), Email (format-validated), Program (dropdown: General Surgery, Orthopedics, Neurosurgery, Cardiac Surgery, Other), Iteration number (1-100), Additional notes
- Input validation happens before submission with clear error dialogs

### Network-Resilient Upload
- Uploads run in a dedicated `VideoSender` QThread with **signal-slot architecture**:
  - `progress_update(int)` → drives a progress bar
  - `upload_complete(str)` → success handling
  - `upload_error(str)` → error handling with retry dialog
  - `connection_status(bool, str)` → live network status indicator
- **Cancel** button with 3-second timeout + force terminate fallback
- **Retry Connection** button that resets retry count and resumes paused uploads
- `NetworkMonitor` class handles periodic connectivity checks

---

## The Server — FastAPI with Async Job Queue

### REST API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/submit` | POST | Submit video + metadata (multipart form) |
| `/status/{id}` | GET | Check submission status (queued/processing/completed/failed) |
| `/queue-status` | GET | Queue length, active jobs, estimated wait time |
| `/submission/{id}` | DELETE | Cancel a queued submission |
| `/health` | GET | Server health check |

### Job Queue — The Core Engine
The `JobQueue` class manages concurrent video processing with several sophisticated mechanisms:

1. **Thread Pool Executor:** Processes up to `MAX_CONCURRENT_JOBS` (default: 3) videos simultaneously using `concurrent.futures.ThreadPoolExecutor`. Video processing runs in threads via `loop.run_in_executor()` to avoid blocking the async event loop
2. **Processing Lock:** An `asyncio.Lock` protects the queue from race conditions during concurrent access
3. **Event-Driven Processing:** A `queue_change_event` (`asyncio.Event`) wakes the processor when new submissions arrive. Between events, the queue checks every 30 seconds for stragglers
4. **Network Awareness:** Before processing, the queue checks connectivity via `NetworkMonitor`. If the network is down, submissions are held — not dropped
5. **Graceful Lifecycle:**
   - Submissions flow through states: `QUEUED` → `PROCESSING` → `COMPLETED` (or `FAILED`, `PENDING_EMAIL`, `EMAIL_FAILED`)
   - Failed email deliveries are automatically retried during periodic maintenance (every 60 seconds)
   - Shutdown is graceful — `shutdown_event` signals the processor, pending jobs complete, thread pool closes

### Email Service
After processing, results are emailed to the trainee's address via SMTP. If the network drops post-processing, the submission moves to `PENDING_EMAIL` status and is retried automatically when connectivity returns.

---

## Edge Cases Engineered For

**Network failures:** Connection timeouts with automatic retries, server unavailability detection, upload interruptions with resume capability, periodic connectivity monitoring on both client and server

**Concurrent submissions:** Thread-safe queue with async locking, configurable parallelism, accurate queue position and ETA calculations

**Resource management:** Video file cleanup after processing, safe camera release on window close (even mid-recording), thread termination with timeout fallback

---

## Stack
`Python` · `FastAPI` · `PyQt5` · `OpenCV` · `AsyncIO` · `ThreadPoolExecutor` · `SMTP` · `Uvicorn`
