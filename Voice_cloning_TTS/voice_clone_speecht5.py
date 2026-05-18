import torchaudio
import huggingface_hub
import traceback
from functools import wraps

# --- Compatibility patches for speechbrain 1.0.3 + newer packages ---
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

_original_hf_hub_download = huggingface_hub.hf_hub_download

@wraps(_original_hf_hub_download)
def _patched_hf_hub_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    try:
        return _original_hf_hub_download(*args, **kwargs)
    except Exception as e:
        called_from_speechbrain = any(
            "speechbrain" in frame.filename for frame in traceback.extract_stack()
        )
        if called_from_speechbrain and ("404" in str(e) or "EntryNotFound" in type(e).__name__):
            raise ValueError(str(e)) from e
        raise

huggingface_hub.hf_hub_download = _patched_hf_hub_download

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
# --- End patches ---

import torch
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wavfile
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


# ========== Step 1: Extract speaker embedding from reference voice ==========
print("Loading speaker encoder (x-vector)...")
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect",
    local_strategy=LocalStrategy.COPY,
)

REF_WAV = "../LJSpeech-1.1/wavs/LJ001-0001.wav"
print(f"Extracting speaker embedding from: {REF_WAV}")

# Load audio and resample to 16 kHz (what the x-vector model expects)
data, sr = sf.read(REF_WAV)
ref_signal = torch.FloatTensor(data)
if sr != 16000:
    print(f"  Resampling from {sr} Hz -> 16000 Hz")
    ref_signal = torchaudio.functional.resample(ref_signal, orig_freq=sr, new_freq=16000)
ref_signal = ref_signal.unsqueeze(0)

custom_embedding = speaker_encoder.encode_batch(ref_signal)
custom_embedding = custom_embedding.squeeze(0)  # shape: [1, 512]
print(f"Custom speaker embedding shape: {custom_embedding.shape}")


# ========== Step 2: Load a KNOWN female x-vector for comparison ==========
# These are the exact embeddings SpeechT5 was designed to use
print("Loading pre-computed CMU ARCTIC x-vectors for comparison...")
xvectors_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Speaker 7306 is a female voice (slt) from the CMU ARCTIC corpus
female_embedding = torch.tensor(xvectors_dataset[7306]["xvector"]).unsqueeze(0)
print(f"Female reference embedding shape: {female_embedding.shape}")


# ========== Step 3: Load SpeechT5 TTS model ==========
print("Loading SpeechT5 TTS model...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# ========== Step 4: Generate speech with BOTH embeddings ==========
text = "This is my voice cloning project using SpeechT5"
print(f'\nGenerating speech: "{text}"')

inputs = processor(text=text, return_tensors="pt")

# --- Output A: Using your custom voice embedding ---
with torch.no_grad():
    speech_custom = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=custom_embedding,
        vocoder=vocoder,
    )
wavfile.write("clone_custom.wav", rate=16000, data=speech_custom.numpy())
print("Saved: clone_custom.wav  (your reference voice embedding)")

# --- Output B: Using the known female CMU ARCTIC embedding ---
with torch.no_grad():
    speech_female = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=female_embedding,
        vocoder=vocoder,
    )
wavfile.write("clone_female_cmu.wav", rate=16000, data=speech_female.numpy())
print("Saved: clone_female_cmu.wav  (known female CMU ARCTIC x-vector)")

print("\n--- Compare both files to see the difference ---")
print("If clone_female_cmu.wav sounds female but clone_custom.wav does not,")
print("it means SpeechT5 cannot faithfully clone arbitrary voices.")
print("Consider using a dedicated voice cloning model like Coqui XTTS-v2 instead.")

