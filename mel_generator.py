import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

wav_path = r"Voice_cloning_TTS/clone_speecht5.wav"

audio, sr = librosa.load(wav_path, sr=22050)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=80
)

mel_db = librosa.power_to_db(mel)

plt.figure(figsize=(8,4))
librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar()
plt.title("Mel Spectrogram")

plt.savefig("mel.png")

plt.show()