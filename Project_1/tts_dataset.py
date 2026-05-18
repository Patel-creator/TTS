import os
import librosa
import torch
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper


# Windows espeak
_espeak_library = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
EspeakWrapper.set_library(_espeak_library)


DATA_PATH = r"D:\TTS\LJSpeech-1.1"
WAV_PATH = os.path.join(DATA_PATH, "wavs")
META_PATH = os.path.join(DATA_PATH, "metadata.csv")


# -----------------------
# text → phoneme
# -----------------------

def text_to_tokens(text):

    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True
    )

    tokens = [ord(c) for c in phonemes]

    return torch.tensor(tokens)


# -----------------------
# audio → mel
# -----------------------

def audio_to_mel(path):

    audio, sr = librosa.load(
        path,
        sr=22050
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80
    )

    mel = torch.tensor(mel)

    return mel


# -----------------------
# load metadata
# -----------------------

def load_dataset():

    samples = []

    with open(META_PATH, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split("|")

            name = parts[0]
            text = parts[2]

            wav = os.path.join(
                WAV_PATH,
                name + ".wav"
            )

            samples.append((wav, text))

    return samples


# -----------------------
# test
# -----------------------

if __name__ == "__main__":

    data = load_dataset()

    wav, text = data[0]

    tokens = text_to_tokens(text)

    mel = audio_to_mel(wav)

    print("Tokens:", tokens.shape)
    print("Mel:", mel.shape)