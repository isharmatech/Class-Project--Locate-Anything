# Locate Anything — Project Website & Live Demo

This repository hosts the project website and the live interactive demo for our DS677 study of **NVIDIA LocateAnything-3B**. The work here is the **static website** (served via GitHub Pages) and the **Hugging Face Space** that embeds the model and powers the live demo.

**Website:** https://isharmatech.github.io/Class-Project--Locate-Anything/
**Live demo:** https://cyberfrost7-locate-anything-demo.hf.space

---

## Short summary of the paper

LocateAnything-3B is a 3B-parameter open-vocabulary vision-language grounding model from NVIDIA. It pairs a **MoonViT-SO-400M** vision encoder with a **Qwen2.5-3B-Instruct** language backbone and replaces the usual token-by-token coordinate generation with **Parallel Box Decoding (PBD)** — predicting an entire bounding box in a single parallel step. This gives up to a **2.5× speedup** over autoregressive approaches (Qwen2.5-VL, Grounding DINO) without losing geometric accuracy, across object detection, phrase grounding, OCR, GUI grounding, and pointing. Full write-up, comparisons, and benchmark tables are on the [website](https://isharmatech.github.io/Class-Project--Locate-Anything/).

---

## What this repository contains

### 1. The website (`index.html` + `assets/`)

A single-page static site that serves as the project writeup and embeds the live demo. Built with plain HTML/CSS/JS (no build step) and deployed through GitHub Pages from the `main` branch.

- **`index.html`** — the full writeup: introduction, architecture, methods, training, inference modes, implementation, demonstration, a "Try It Yourself" section with the embedded demo, related work, quantitative benchmarks, limitations & future work, conclusion, and references.
- **`assets/style.css`** — theming (light/dark via CSS variables, display + body font stacks, responsive layout).
- **`assets/main.js`** — theme toggle, scroll reveal, and the **lazy-loaded Gradio iframe** with a graceful fallback.
- **`assets/config.js`** — one-line config for the live demo URL and iframe load timeout.
- **`assets/images/`** — architecture diagrams and demo output samples.

**Embedded demo behavior:** the Gradio iframe lazy-loads only when the "Try It Yourself" section approaches the viewport. If the Hugging Face Space is asleep or the iframe fails to load, the site shows a **"The demo Space is asleep"** fallback card with a button to open the Space directly, then prompts the visitor to refresh the page once it's awake.

### 2. The Hugging Face Space (`hf-space/`)

A persistent, ZeroGPU-backed Gradio app that **embeds `nvidia/LocateAnything-3B`** so anyone can run the model in the browser. This is what the website embeds.

- **`hf-space/app.py`** — the Gradio app adapted for ZeroGPU: the model loads on CPU at startup and moves onto the GPU inside an `@spaces.GPU`-decorated `predict` step. The UI presents a gallery of example images (Step 1), then task/description/inference-mode controls and the result panels (Step 2). Image upload is disabled (`interactive=False`) so visitors pick from the bundled examples.
- **`hf-space/requirements.txt`** — pinned deps for the Space runtime.
- **`hf-space/README.md`** — Space metadata (Gradio SDK, ZeroGPU hardware, Python version, tags).
- **`hf-space/assets/`** — the example images bundled with the Space.

### 3. Deployment helpers

- **`upload_space.py`** — one-shot script that creates the Space (if missing), uploads `hf-space/`, and sets the `HF_TOKEN` Space secret so ZeroGPU attributes GPU-minutes to your Pro account instead of anonymous IP-based quotas.
- **`colab_demo.ipynb`** — a Colab alternative for an ephemeral T4 share-link demo.
- **`app.py`** + **`requirements.txt`** — the local Gradio app used as the base for both the Colab and Space builds.

---

## Repository Structure

```
.
├── index.html                  # The website (GitHub Pages writeup + embedded demo)
├── assets/                     # Site CSS/JS, config, demo images
│   ├── config.js               # Live demo URL + iframe fallback settings
│   ├── main.js                 # Lazy-load iframe + "Space is asleep" fallback
│   ├── style.css               # Site styling
│   └── images/                 # Architecture diagrams + demo output samples
├── hf-space/                   # Hugging Face Space source (ZeroGPU)
│   ├── app.py                  # ZeroGPU-adapted Gradio app that embeds the model
│   ├── requirements.txt        # Space deps
│   ├── README.md               # Space metadata
│   └── assets/                 # Example images bundled with the Space
├── app.py                      # Local Gradio app (base for Colab + Space)
├── requirements.txt            # Pinned Python deps for the local app
├── colab_demo.ipynb            # Colab notebook: T4 GPU, share-link demo
├── upload_space.py             # Create/upload the HF Space + set HF_TOKEN secret
├── LocateAnything_Demo.ipynb   # Original experiment notebook
└── DS677 Project_7_25.docx     # Project brief
```

## Run the website locally

Because it's static, you can preview it with any static server:

```bash
# Python
python -m http.server 8000
# then open http://localhost:8000
```

Edit `assets/config.js` to point `GRADIO_LIVE_URL` at your own Space or share link.

## Run the local Gradio app

```bash
pip install -r requirements.txt
python app.py
```

A CUDA GPU with ≥10 GB VRAM is recommended; the model loads in fp16.

## Deploy / redeploy the Hugging Face Space

```bash
python upload_space.py
# paste a Hugging Face write token when prompted
```

This uploads `hf-space/` to `cyberfrost7/locate-anything-demo` and sets the `HF_TOKEN` secret for ZeroGPU quota attribution. The Space URL is persistent — no demo-day scramble to update links.

## Run the Colab demo

Open `colab_demo.ipynb`, set the runtime to **T4 GPU**, confirm with `!nvidia-smi`, then run all cells. It installs pinned deps, loads `nvidia/LocateAnything-3B` in fp16, and launches Gradio with a public `*.gradio.live` share link (valid while the session runs).
