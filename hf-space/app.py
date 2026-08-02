"""Locate Anything 3B — Hugging Face Space app (ZeroGPU).

Adapted from the project repo's app.py. The only difference is ZeroGPU
scheduling: the model loads onto CPU at startup and is moved to the GPU
inside @spaces.GPU on the first inference call. The Gradio UI (build_demo)
is identical to the local/Colab version so the demo looks the same everywhere.
"""
import os
import re
from typing import Any

import gradio as gr
import spaces  # ZeroGPU — preinstalled on Gradio SDK Spaces
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
HF_TOKEN = os.getenv("HF_TOKEN")

# ZeroGPU allocates the GPU only inside @spaces.GPU functions, so we load the
# model onto CPU at startup and move it to cuda on the first inference call.
# fp16 is universally safe; Blackwell also supports bf16.
DTYPE = torch.float16


class LocateAnythingWorker:
    """Loads LocateAnything once and serves repeated Gradio requests."""

    def __init__(self, model_id: str) -> None:
        common_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if HF_TOKEN:
            common_kwargs["token"] = HF_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, **common_kwargs)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            **common_kwargs,
        ).eval()
        self._on_gpu = False

    @spaces.GPU
    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        # Move the model to the ZeroGPU-allocated GPU once; it stays resident
        # for the lifetime of this Space container (ZeroGPU keeps the GPU
        # attached across @spaces.GPU calls within the same warm container).
        if not self._on_gpu:
            self.model = self.model.to("cuda")
            self._on_gpu = True

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to("cuda")

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(DTYPE),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            generation_mode=generation_mode,
            temperature=float(temperature),
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )

        answer = response[0] if isinstance(response, tuple) else response
        return str(answer)


_WORKER: LocateAnythingWorker | None = None


def get_worker() -> LocateAnythingWorker:
    """Return the shared LocateAnything worker, loading it on first call."""
    global _WORKER
    if _WORKER is None:
        _WORKER = LocateAnythingWorker(MODEL_ID)
    return _WORKER


def resize_for_demo(image: Image.Image, max_side: int) -> Image.Image:
    """Limit image size to reduce GPU-memory usage while preserving aspect ratio."""
    image = image.convert("RGB")
    if not max_side or max(image.size) <= max_side:
        return image

    width, height = image.size
    scale = max_side / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def build_prompt(task: str, description: str) -> str:
    description = (description or "").strip()

    if task == "Object Detection":
        categories = [item.strip() for item in description.split(",") if item.strip()]
        if not categories:
            raise gr.Error("Enter at least one category, such as: person, car, bus")
        joined = "</c>".join(categories)
        return f"Locate all the instances that matches the following description: {joined}."

    if task == "Phrase Grounding":
        if not description:
            raise gr.Error("Enter a phrase, such as: the person wearing a red shirt")
        return f"Locate all the instances that match the following description: {description}."

    if task == "OCR / Text Detection":
        return "Detect all the text in box format."

    if task == "GUI Grounding":
        if not description:
            raise gr.Error("Enter a GUI element, such as: the search button")
        return f"Locate the region that matches the following description: {description}."

    if task == "Pointing":
        if not description:
            raise gr.Error("Enter an item to point to, such as: the traffic light")
        return f"Point to: {description}."

    raise gr.Error(f"Unsupported task: {task}")


def parse_labeled_boxes(answer: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Extract normalized [0,1000] box coordinates and nearby labels."""
    pattern = re.compile(
        r"(?:<ref>(.*?)</ref>\s*)?<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        flags=re.DOTALL,
    )
    results: list[tuple[str, tuple[int, int, int, int]]] = []
    for match in pattern.finditer(answer):
        label = re.sub(r"\s+", " ", (match.group(1) or "object")).strip()
        coords = tuple(int(match.group(i)) for i in range(2, 6))
        results.append((label, coords))
    return results


def parse_points(answer: str) -> list[tuple[int, int]]:
    """Extract normalized [0,1000] point coordinates."""
    return [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer)
    ]


def annotate_image(image: Image.Image, answer: str) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    width, height = output.size
    line_width = max(2, round(min(width, height) / 250))

    for label, (x1, y1, x2, y2) in parse_labeled_boxes(answer):
        left = max(0, min(width - 1, round(x1 / 1000 * width)))
        top = max(0, min(height - 1, round(y1 / 1000 * height)))
        right = max(0, min(width - 1, round(x2 / 1000 * width)))
        bottom = max(0, min(height - 1, round(y2 / 1000 * height)))

        draw.rectangle((left, top, right, bottom), outline="red", width=line_width)

        text_box = draw.textbbox((left, top), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, top - text_height - 6)
        draw.rectangle(
            (left, label_top, min(width - 1, left + text_width + 8), top),
            fill="red",
        )
        draw.text((left + 4, label_top + 2), label, fill="white", font=font)

    radius = max(5, round(min(width, height) / 100))
    for x, y in parse_points(answer):
        px = max(0, min(width - 1, round(x / 1000 * width)))
        py = max(0, min(height - 1, round(y / 1000 * height)))
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            outline="red",
            width=line_width,
        )
        draw.line((px - radius, py, px + radius, py), fill="red", width=line_width)
        draw.line((px, py - radius, px, py + radius), fill="red", width=line_width)

    return output


def run_inference(
    image: Image.Image | None,
    task: str,
    description: str,
    generation_mode: str,
    max_side: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[Image.Image, str, str]:
    if image is None:
        raise gr.Error("Upload an image first.")

    prepared_image = resize_for_demo(image, int(max_side))
    prompt = build_prompt(task, description)
    answer = get_worker().predict(
        image=prepared_image,
        prompt=prompt,
        generation_mode=generation_mode,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )
    annotated = annotate_image(prepared_image, answer)
    return annotated, prompt, answer


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI — identical to the local/Colab app.py."""
    with gr.Blocks(title="Locate Anything Demo") as demo:
        gr.Markdown(
            "# Locate Anything Demo\n"
            "Upload an image and describe what you want NVIDIA LocateAnything-3B to locate."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input image")
                task_input = gr.Dropdown(
                    choices=[
                        "Object Detection",
                        "Phrase Grounding",
                        "OCR / Text Detection",
                        "GUI Grounding",
                        "Pointing",
                    ],
                    value="Object Detection",
                    label="Task",
                )
                description_input = gr.Textbox(
                    value="person, car, bus",
                    label="What should the model locate?",
                    placeholder="Examples: person, car, bus OR the person wearing red",
                )
                mode_input = gr.Radio(
                    choices=["hybrid", "fast", "slow"],
                    value="hybrid",
                    label="Inference mode",
                )

                with gr.Accordion("Advanced settings", open=False):
                    max_side_input = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=128,
                        label="Maximum image side",
                    )
                    max_tokens_input = gr.Slider(
                        minimum=512,
                        maximum=8192,
                        value=2048,
                        step=512,
                        label="Maximum new tokens",
                    )
                    temperature_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )

                run_button = gr.Button("Locate objects", variant="primary")

                # Pre-loaded example images so visitors can try the demo without
                # uploading their own. Each example auto-fills the image, task, and
                # description to reproduce the experiments from the project report.
                gr.Examples(
                    examples=[
                        ["assets/street_scene.jpg", "Object Detection", "person, car, bicycle", "hybrid"],
                        ["assets/desk_scene.jpg", "Object Detection", "computer, mouse, cup, keyboard, plant", "hybrid"],
                        ["assets/grocery_shelf.jpg", "Phrase Grounding", "the blue and yellow pasta box labeled Whole Wheat Elbows", "hybrid"],
                        ["assets/clothing_store.jpg", "Object Detection", "dress, shirt, jacket, clothing rack, handbag", "hybrid"],
                    ],
                    inputs=[image_input, task_input, description_input, mode_input],
                    label="Examples — click one to load it, then press Locate objects",
                    # Don't pre-compute/run at startup — that would burn ZeroGPU quota.
                    # Clicking an example only fills the inputs; the user then hits the button.
                    cache_examples=False,
                )

            with gr.Column():
                output_image = gr.Image(type="pil", label="Detected result")
                generated_prompt = gr.Textbox(label="Prompt sent to model", lines=2)
                raw_output = gr.Textbox(label="Raw model output", lines=8)

        run_button.click(
            fn=run_inference,
            inputs=[
                image_input,
                task_input,
                description_input,
                mode_input,
                max_side_input,
                max_tokens_input,
                temperature_input,
            ],
            outputs=[output_image, generated_prompt, raw_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch()
