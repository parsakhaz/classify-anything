# Image Classification with Moondream and LLaMA

> **⚠️ IMPORTANT:** This project currently uses Moondream 2B (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client libraries once they become available for this version.
>
> **⚠️ NOTE:** This project requires access to Meta's LLaMA 3.2 3B Instruct model via HuggingFace. You must request and be granted access before using this script. Visit [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) to request access.
>
> **⚠️ AUTHENTICATION:** When using HuggingFace authentication, make sure to use a token with "WRITE" permission, not "FINEGRAINED" permission. If you encounter authorization issues despite having model access, generate a new token with "WRITE" permission.

## Table of Contents
- [Overview](#overview)
- [Sample Output](#sample-output)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Options](#command-line-options)
  - [Examples](#examples)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Dependencies](#dependencies)
- [Model Details](#model-details)
- [License](#license)

## Overview
This project automatically analyzes images using AI to classify multiple aspects simultaneously. It combines LLaMA for intelligent question formulation and Moondream for visual analysis, providing detailed insights about various aspects of images.

## Sample Output
```
Processing: park.jpg
======================================================================
Final Classification Results:
------------------------------
Aspect: grass color
Result: Green
------------------------------
Aspect: time of day
Result: Afternoon
------------------------------
Aspect: number of people, if any
Result: 0
------------------------------
Aspect: weather conditions
Result: Sunny
------------------------------
Aspect: main activity
Result: A winding path lined with trees
------------------------------
```

## Features
- **Multi-Aspect Classification**:
  - Customizable aspects to analyze
  - Default aspects include:
    1. Grass color
    2. Time of day
    3. Number of people
    4. Weather conditions
    5. Main activity/focus

- **Intelligent Question Generation**:
  - Uses LLaMA to formulate natural questions
  - Context-aware question formatting
  - Optimized for concise answers

- **Efficient Processing**:
  - Batch processing of multiple images
  - Smart memory management
  - GPU optimization

- **Flexible Input Handling**:
  - Supports multiple image formats
  - Processes entire directories
  - Error handling for unsupported files

## Prerequisites
1. Python 3.8 or later
2. CUDA-capable GPU (8GB+ VRAM recommended)
3. HuggingFace account with approved access to Meta's LLaMA 3.2 3B Instruct model

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd classify-anything
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

### System Dependencies
```bash
# Linux/Ubuntu
sudo apt-get update
sudo apt-get install libvips libvips-dev

# macOS with Homebrew
brew install vips

# Windows
# Download and install libvips from https://github.com/libvips/build-win64/releases
```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your images in the `inputs` folder
2. Authenticate with HuggingFace:
   ```bash
   huggingface-cli login
   ```

3. Run the script:
   ```bash
   python classify.py
   ```

### Command-Line Options
[Future enhancement: Add command-line options for customizing aspects to classify]

### Examples
```bash
# Process all images in inputs directory
python classify.py

# [Future enhancement: Add examples with custom options]
```

## Output
The script provides:
1. **Console Output**:
   - Processing status for each image
   - Generated questions
   - Classification results
   - Error messages (if any)

2. **Results Format**:
   - Aspect-by-aspect breakdown
   - Concise, clear answers
   - Structured presentation

## Troubleshooting
1. CUDA/GPU Issues:
   - Ensure CUDA is properly installed
   - Verify GPU has sufficient VRAM (8GB+ recommended)
   - Close other GPU-intensive applications

2. Memory Issues:
   - Process fewer images at once
   - Clear GPU cache between runs
   - Monitor system resources

3. Model Loading Issues:
   - Verify HuggingFace authentication
   - Check LLaMA model access status
   - Update transformers library
   - **Important**: Use a HuggingFace token with "WRITE" permission, not "FINEGRAINED" permission
   - If you encounter authorization issues despite having model access, try generating a new token with "WRITE" permission

## Performance Notes
- Processing time depends on:
  - Number of images
  - Number of aspects to classify
  - GPU memory and speed

- Memory Usage:
  - Moondream model: ~4GB VRAM
  - LLaMA model: ~6GB VRAM
  - Efficient memory management between model loads

## Dependencies
Required Python packages:
- transformers
- torch
- Pillow (PIL)
- opencv-python (cv2)
- huggingface-hub
- accelerate

All dependencies can be installed via:
```bash
pip install -r requirements.txt
```

## Model Details
- **Question Generation**: Meta's LLaMA 3.2 3B Instruct
  - Generates natural language questions
  - Optimized for clear, focused queries

- **Image Analysis**: Moondream 2B (2025-01-09 release)
  - Specialized for detailed image understanding
  - Provides concise, accurate answers

## License
This project is licensed under the MIT License. 