import torchaudio
import huggingface_hub
from functools import wraps

# Fix 1: speechbrain expects list_audio_backends which may not exist in this torchaudio version
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# Fix 2: newer huggingface_hub removed 'use_auth_token' and raises
# RemoteEntryNotFoundError instead of ValueError for missing files.
# speechbrain 1.0.3 expects ValueError to gracefully skip missing custom.py.
_original_hf_hub_download = huggingface_hub.hf_hub_download
@wraps(_original_hf_hub_download)
def _patched_hf_hub_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    try:
        return _original_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            raise ValueError(str(e)) from e
        raise
huggingface_hub.hf_hub_download = _patched_hf_hub_download

from speechbrain.inference.speaker import EncoderClassifier


from speechbrain.utils.fetching import LocalStrategy

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models",
    local_strategy=LocalStrategy.COPY,
)


import soundfile as sf
import torch

data, fs = sf.read("../LJSpeech-1.1/wavs/LJ001-0001.wav")
signal = torch.FloatTensor(data).unsqueeze(0)  # add batch/channel dim


embedding = classifier.encode_batch(signal)


print("Embedding shape:", embedding.shape)