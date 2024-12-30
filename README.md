## MuseTalk Superresolution Enhancement
# Overview
This repository is an enhanced version of the MuseTalk project, designed to improve the resolution of the lipsynced region in the generated videos.
The enhancement is achieved by integrating two state-of-the-art superresolution models: 
GFPGAN and CodeFormer.

# Key Features
Superresolution
Boosts the resolution of the lipsynced region while preserving the quality of the original frame.
Supports GFPGAN and CodeFormer for superresolution.
# Customizable
Users can select their preferred superresolution model using the --superres argument.
Ease of Use
Simple command-line interface for processing videos.
Setup Instructions
Step 1: Clone the Repository
bash

# Clone the MuseTalk repository
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

# Clone this enhanced repository
# Replace with your GitHub repository URL after pushing the project
Step 2: Install Python & Dependencies on Windows
Install Python
Download Python 3.8+ from the official Python website.
Make sure to select "Add Python to PATH" during installation.
Verify installation:
bash
python --version
pip --version
Install Dependencies
Run the following commands in the VS Code terminal:

bash
pip install torch torchvision moviepy opencv-python argparse tqdm

# Install GFPGAN dependencies
pip install basicsr facexlib realesrgan

# Install CodeFormer dependencies
pip install -r CodeFormer/requirements.txt
Step 3: Download Pre-trained Models
# GFPGAN Model
Download the GFPGAN pre-trained model from here.
Place the downloaded model in the 
GFPGAN/experiments/pretrained_models/ directory.
# CodeFormer Model
Download the CodeFormer pre-trained model from here.
Place the downloaded model in the CodeFormer/weights/ directory.

Step 4: Prepare a Test Video
Prepare a 3-second test video with a visible face for testing. 
Ensure the video file is in the same directory or specify its absolute path in commands.

# Usage Instructions
Run the script x.py using the following command:

bash
python x.py --superres [GFPGAN/CodeFormer] -iv input_video.mp4 -ia input_audio.mp3 -o output_video.mp4

Arguments
--superres: Choose the superresolution method (GFPGAN or CodeFormer).
-iv: Path to the input video.
-ia: Path to the input audio.
-o: Path to the output video.
Example Command

bash
python x.py --superres GFPGAN -iv input.mp4 -ia input.mp3 -o output.mp4

# Project Workflow
Load Video and Audio:

The script processes the video frame by frame.
The lipsynced region is identified and extracted.
Resolution Ratio Calculation:

The resolution of the original frame and the generated subframe are compared.
If the generated subframe has a lower resolution, the superresolution model is applied.
Apply Superresolution:

The user-selected model (GFPGAN or CodeFormer) enhances the low-resolution region.
The enhanced region is merged back into the original frame.
Save the Output:

The processed frames are combined into a new video.
# Results
The output video will have a high-resolution lipsynced region, improving the overall quality of the generated content.

# Troubleshooting
Error: Missing Pre-trained Model
Ensure the GFPGAN and CodeFormer pre-trained models are correctly downloaded and placed in their respective directories.
# ModuleNotFoundError
Check that all dependencies are installed using the provided commands.
# Slow Processing
The script is designed for short videos (3 seconds) to avoid delays. Ensure your input video is of similar length.
# Contributing
If you encounter any issues or have suggestions, feel free to create a pull request or open an issue on GitHub.

# License
This project inherits the licenses of MuseTalk, GFPGAN, and CodeFormer. Refer to their respective repositories for details.

Developer Information
Khushi Mittal
Email: khushimittal2811@gmail.com

## MuseTalk: Real-Time High-Quality Lip Synchronization with Latent Space Inpainting
This project introduces MuseTalk, a real-time high-quality lip-syncing model trained in the latent space of ft-mse-vae, which:

Modifies an unseen face according to the input audio (face region of 256 x 256).
Supports audio in various languages (Chinese, English, Japanese).
Provides real-time inference (30fps+ on an NVIDIA Tesla V100).
Installation
Building the Environment
We recommend Python version >=3.10 and CUDA version 11.7. Then, set up your environment by installing the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
Install mmlab Packages
bash
Copy code
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
Download ffmpeg-static
bash
Copy code
export FFMPEG_PATH=/path/to/ffmpeg
Download Weights
Download the necessary weights:

MuseTalk model weights
sd-vae-ft-mse
whisper
dwpose
face-parse-bisent
Inference Quickstart
Run inference with:

bash
Copy code
python -m scripts.inference --inference_config configs/inference/test.yaml 
To adjust the bbox_shift:

bash
Copy code
python -m scripts.inference --inference_config configs/inference/test.yaml --bbox_shift -7
Video Examples & Cases
MuseTalk can animate static images and apply dubbing to video content, as shown in the demo cases.

TODO List:
 Trained models and inference codes.
 Huggingface Gradio demo.
 Technical report.
 Training codes.
 Improved model (may take longer).
