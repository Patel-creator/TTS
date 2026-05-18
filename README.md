# TTS Voice Cloning Project

This project contains various scripts and Jupyter notebooks for Text-to-Speech (TTS) and voice cloning using models like `SpeechT5` and `Coqui XTTS-v2`.

## Prerequisites

1. **Python**: Python 3.9+ is recommended.
2. **System Dependencies (eSpeak NG)**:
   This project relies on `phonemizer` which requires `eSpeak NG` to be installed on your system.
   - **Windows**: Download and install from [eSpeak NG GitHub](https://github.com/espeak-ng/espeak-ng/releases). By default, it installs to `C:\Program Files\eSpeak NG\libespeak-ng.dll`. If installed elsewhere, set the `ESPEAK_LIBRARY` environment variable.
   - **Linux**: `sudo apt-get install espeak-ng`
   - **Mac**: `brew install espeak`

3. **LJSpeech Dataset**:
   Several scripts in `Project_1/` and `Voice_cloning_TTS/` rely on the LJSpeech dataset.
   - Download it from [here](https://keithito.com/LJ-Speech-Dataset/).
   - Extract it so that the `LJSpeech-1.1/` directory is in the root of this project (at the same level as `Voice_cloning_TTS/`).

## Installation

To run this project on a new device, you should set up a virtual environment and install the required dependencies.

```bash
# Create a virtual environment
python -m venv tts_env

# Activate it (Windows)
tts_env\Scripts\activate
# Activate it (Linux/Mac)
source tts_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: PyTorch dependencies in `requirements.txt` might be specific to the original system architecture. If you encounter issues, install PyTorch/Torchaudio manually according to your system from [pytorch.org](https://pytorch.org/).)*

## Usage

You can explore the different components:
- **`Voice_cloning_TTS/xtts_clone.py`**: Uses Coqui XTTS-v2 to clone a voice from a reference audio file (`../LJSpeech-1.1/wavs/LJ001-0001.wav` by default).
- **`Voice_cloning_TTS/voice_clone_speecht5.py`**: Voice cloning using SpeechT5 and X-Vectors.
- **`Project_1/`**: Contains scripts to process the LJSpeech dataset and generate Mel-spectrograms.
- **Notebooks (`first.ipynb`, `day2.ipynb`)**: Interactive environments for exploring TTS processing.

## Important Note

Models like XTTS-v2 and SpeechT5 will download their weights automatically from Hugging Face the first time you run them. This may take some time depending on your internet connection.
