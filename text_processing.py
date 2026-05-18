# text_processing.py

from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper


import os

# -------------------------
# Windows espeak path
# -------------------------

_espeak_library = os.environ.get("ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
if os.path.exists(_espeak_library):
    EspeakWrapper.set_library(_espeak_library)


# -------------------------
# 1. normalize
# -------------------------
def normalize_text(text):
    return text.lower()


# -------------------------
# 2. text → phoneme
# -------------------------
def text_to_phoneme(text):

    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True
    )

    return phonemes


# -------------------------
# 3. phoneme → tokens
# -------------------------
def phoneme_to_tokens(phonemes):

    tokens = [ord(c) for c in phonemes]

    return tokens


# -------------------------
# main
# -------------------------
if __name__ == "__main__":

    text = input("Enter text: ")

    text = normalize_text(text)
    print("\nNormalized:", text)

    phonemes = text_to_phoneme(text)
    print("\nPhonemes:", phonemes)

    tokens = phoneme_to_tokens(phonemes)
    print("\nTokens:", tokens)