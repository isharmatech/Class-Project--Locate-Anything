# Gradio implementation for Locate-Anything-Demo

## Files to add to the GitHub repository

Add these files at the repository root:

- `app.py`
- `requirements.txt`

The app loads `nvidia/LocateAnything-3B` once, accepts an uploaded image and a natural-language target, runs the selected grounding task, parses the model's normalized coordinates, and draws bounding boxes or points on the output image.

## Run locally

A CUDA-capable NVIDIA GPU is strongly recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio address printed in the terminal.

## Run in Google Colab

After installing the requirements and running the model-loading code, change the last line to:

```python
demo.queue().launch(share=True, debug=True)
```

Use a GPU runtime. The first execution downloads the model weights.

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Upload `app.py` and `requirements.txt`.
4. Select GPU hardware in the Space settings.
5. Add an `HF_TOKEN` secret only if the model download requires authentication.

## Important implementation details

- Default inference mode is `hybrid`.
- Images are resized to a maximum side of 1024 pixels by default to reduce GPU-memory usage.
- LocateAnything returns normalized coordinates from 0 to 1000; the app converts them to pixel coordinates before drawing.
- The NVIDIA model license should be reviewed before publishing or reusing the application.
