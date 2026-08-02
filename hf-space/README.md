---
title: Locate Anything 3B Demo
emoji: 🎯
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: "6.16.0"
python_version: "3.12"
app_file: app.py
hardware:
  - zero-gpu
models:
  - nvidia/LocateAnything-3B
tags:
  - visual-grounding
  - object-detection
  - vision-language
  - locateanything
short_description: Live visual-grounding demo of NVIDIA LocateAnything-3B
---

# Locate Anything 3B — Live Demo

Upload an image, choose a grounding task, describe what you want to locate, and
NVIDIA **LocateAnything-3B** draws bounding boxes (or points) on the matching
objects. Runs on Hugging Face ZeroGPU (NVIDIA RTX Pro 6000 Blackwell).

**Tasks:** Object Detection · Phrase Grounding · OCR / Text Detection · GUI
Grounding · Pointing.

**Inference modes:** `hybrid` (default, Parallel Box Decoding + autoregressive
fallback), `fast` (pure parallel decoding), `slow` (autoregressive).

The first request after the Space wakes up takes ~30–60 s while the 3B checkpoint
loads onto the GPU; subsequent requests are fast.
