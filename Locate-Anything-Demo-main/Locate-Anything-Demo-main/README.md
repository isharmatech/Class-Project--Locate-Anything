## Locate-Anything-Demo

This project demonstrates NVIDIA's **LocateAnything-3B**, a vision-language grounding model that locates objects in images using natural-language prompts.

The application includes a Gradio interface that allows users to upload an image, describe what they want to locate, and view the detected objects with bounding boxes or points.

## Features

* Object detection
* Phrase grounding
* OCR and text detection
* GUI element grounding
* Object pointing
* Image annotations with bounding boxes
* Raw model output display
* Multiple inference modes

## Project Structure

```text
Locate-Anything-Demo/
├── app.py
├── requirements.txt
├── README.md
├── Locate Anything Demo/
├── Original Images/
└── Detected Images/
```

## Requirements

Before running the application, make sure you have:

* Python 3.11
* Git
* An internet connection for the initial model download
* An NVIDIA CUDA-capable GPU is strongly recommended

The application can run on a CPU, but inference may be very slow.

## Run the Program on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/ella-mahalia/Locate-Anything-Demo.git
cd Locate-Anything-Demo
```

### 2. Create a Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should begin with:

```text
(.venv)
```

### 4. Upgrade the installation tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install the required packages

```powershell
python -m pip install -r requirements.txt
```

### 6. Start the Gradio application

```powershell
python app.py
```

After the model loads, Gradio will display a local address similar to:

```text
http://127.0.0.1:7860
```

Open that address in your web browser.

## Run the Program on macOS or Linux

```bash
git clone https://github.com/ella-mahalia/Locate-Anything-Demo.git
cd Locate-Anything-Demo

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

## How to Use the Application

1. Upload an image.
2. Select a task.
3. Enter the object, category, or phrase you want the model to locate.
4. Select an inference mode.
5. Click **Locate Objects**.
6. Review the annotated image and raw model response.

For object detection, enter categories separated by commas:

```text
person, car, bus
```

For phrase grounding, enter a descriptive phrase:

```text
the person wearing a red shirt
```

## Available Tasks

### Object Detection

Locates all instances of one or more categories.

### Phrase Grounding

Locates an object or region that matches a detailed natural-language description.

### OCR / Text Detection

Detects visible text regions in an image.

### GUI Grounding

Locates interface elements such as buttons, menus, and search fields.

### Pointing

Returns a point representing the location of a requested object.

## Troubleshooting

### `ModuleNotFoundError: No module named 'gradio'`

Make sure the virtual environment is activated, and then reinstall the requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Installation fails with Python 3.13

This project uses Python 3.11. Confirm the active Python version:

```powershell
python --version
```

The result should begin with:

```text
Python 3.11
```

### The model is running slowly

A CUDA-capable NVIDIA GPU is recommended. CPU inference may take significantly longer.

You can also reduce the maximum image size in the application's advanced settings.

### The first launch takes a long time

The application downloads the LocateAnything-3B model during the first launch. Later launches can reuse the downloaded model files.

## Model

This project uses:

```text
nvidia/LocateAnything-3B
```

Review the model's license and usage conditions before deploying or redistributing the application.
