import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from TTS.api import TTS

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
)

tts.tts_to_file(
    text="This is my voice cloning project",
    speaker_wav="../LJSpeech-1.1/wavs/LJ001-0001.wav",
    language="en",
    file_path="clone.wav"
)