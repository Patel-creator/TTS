import torch

# fake tokens (from day2)
tokens = torch.randint(0, 50, (1, 10))

print("Tokens shape:", tokens.shape)

# fake encoder output
encoder_out = torch.randn(1, 10, 128)

print("Encoder:", encoder_out.shape)

# fake decoder output (mel spectrogram)
mel = torch.randn(1, 80, 100)

print("Mel:", mel.shape)