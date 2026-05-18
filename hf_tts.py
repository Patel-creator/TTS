import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile


model = VitsModel.from_pretrained(
    "facebook/mms-tts-eng"
)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-eng"
)


text = "I am a god!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform


audio = output.squeeze().numpy()

wavfile.write(
    "output.wav",   
    rate=model.config.sampling_rate,
    data=audio
)

print("Saved output.wav")