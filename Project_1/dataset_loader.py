import os
import librosa
import pandas as pd


DATA_PATH = r"D:\TTS\LJSpeech-1.1"
WAV_PATH = os.path.join(DATA_PATH, "wavs")
META_PATH = os.path.join(DATA_PATH, "metadata.csv")


def load_metadata():

    data = []

    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            name, text, _ = line.strip().split("|")

            wav_file = os.path.join(
                WAV_PATH,
                name + ".wav"
            )

            data.append((wav_file, text))

    return data


def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=22050
    )

    return audio, sr


def mel_spectrogram(audio, sr):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80
    )

    return mel


if __name__ == "__main__":

    data = load_metadata()

    for wav, text in data:

        print("Text:", text)

        audio, sr = load_audio(wav)

        mel = mel_spectrogram(audio, sr)

        print("Mel shape:", mel.shape)

        break
