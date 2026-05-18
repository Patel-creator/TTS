import torch
import scipy.io.wavfile as wavfile

from transformers import VitsModel, AutoTokenizer


model = VitsModel.from_pretrained(
    "facebook/mms-tts-eng"
)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-eng"
)


text = "This speech is generated using my voice cloning project"


inputs = tokenizer(
    text,
    return_tensors="pt"
)


with torch.no_grad():
    output = model(**inputs).waveform


audio = output.squeeze().numpy()


wavfile.write(
    "demo.wav",
    rate=model.config.sampling_rate,
    data=audio
)

print("saved demo.wav")