---
title: "Job Assistant"
date: 2026-07-15
draft: false
projectType: "Side Projects"
tags:
- Job Search
- Assistant
description: "Autonomous multi-agent system that applies for jobs end-to-end — scraping, evaluating, and form-filling across ATS platforms using a Llama 3 + Gemini dual-brain architecture for $0 local compute."
---

## TL;DR
AutoJob is a highly autonomous multi-agent system designed to scrape the internet for job postings, evaluate them against a user's resume, and execute full application workflows (including multi-page forms and account creation). It solves the compute-cost constraints of traditional web agents by utilizing a decoupled architecture.

## The Web Automation Bottleneck
Traditional web agents (like AutoGPT) rely entirely on massive cloud-based LLMs (e.g., GPT-4) for every single decision. When navigating complex, modern web applications (like Workday or Greenhouse), the token payload of massive HTML DOMs paired with a user's resume and codebase context easily exceeds millions of tokens per job. This approach costs hundreds of dollars a day and quickly exhausts context windows, making large-scale automation financially unviable.

## Breaking Down the Cognitive Load
The goal was to build a system that achieves human-level autonomy without the astronomical cloud costs. By breaking down the cognitive workload into its smallest sensible parts, we realize that not every task requires a massive cloud LLM. Answering technical questions based on a resume requires deep semantic reasoning but no spatial awareness. Clicking a button requires spatial awareness but no deep semantic reasoning. Therefore, the most efficient solution is an asymmetric compute model that delegates tasks to the cheapest capable processor.

## Decoupled Asymmetric Compute

The system is built on a **Decoupled Asymmetric Compute Architecture**, orchestrating two distinct "brains" via a LangGraph state machine:

### 1. The Semantic Brain (Local Llama 3 8B)
Running locally on an NVIDIA GTX 1650 Ti for $0 cost, this brain handles privacy-sensitive and knowledge-heavy tasks:
- **Retrieval-Augmented Generation (RAG)**: Uses `sentence-transformers` and ChromaDB to ingest the user's resume and local codebases. If an application asks a technical question, the Semantic Brain queries the local vector database and drafts an essay quoting actual code.
- **The Critic (Safety Validator)**: Acts as an interceptor before any web action is executed. It evaluates proposed actions ("Is it safe to type 'John' here?") to prevent LLM hallucinations and agent death spirals on complex dropdowns.

### 2. The Spatial Brain (Cloud Gemini 2.5 Flash)
Operating on the cloud, this brain acts as the spatial navigator. It receives a heavily compressed version of the website's DOM (stripped of scripts, styles, and SVGs) and maps the Semantic Brain's intents to CSS selectors for execution.

### Web Execution & Hardening
The physical actions are executed via Playwright, hardened against real-world ATS defenses:
- **DOM Compression & Shadow DOMs**: Iterates over `page.frames` to extract elements from nested Cross-Origin iFrames, bypassing ATS obfuscation.
- **Bot Evasion**: Utilizes `playwright-stealth`, simulates human keystrokes (`.press_sequentially`), and forcibly bypasses floating GDPR cookie banners.
- **2FA Injection**: An integrated IMAP utility (`email_reader.py`) polls Gmail to extract 6-digit verification codes, bridging the gap when forced to create dummy accounts.
