import torch
from src.models.decoder import GeneratorS1, GeneratorS2
import types

# Config fake per test rapido
class DummyConfig:
    TEXT_EMBEDDING_DIM = 256
    Z_DIM = 100
    DECODER_BASE_CHANNELS = 128
    NUM_HEADS = 8
    STAGE2_IMAGE_SIZE = 215

config = DummyConfig()

# Simula batch
batch_size = 2
seq_len = 16

# Simula input per GeneratorS1
cls_embedding = torch.randn(batch_size, config.TEXT_EMBEDDING_DIM)
hidden_states = torch.randn(batch_size, seq_len, config.TEXT_EMBEDDING_DIM)
z_noise = torch.randn(batch_size, config.Z_DIM)

# Istanzia e testa GeneratorS1
netG1 = GeneratorS1(config)
img1, attn = netG1(cls_embedding, hidden_states, z_noise)
print(f"GeneratorS1 output: {img1.shape}")

# Simula input per GeneratorS2
stage1_mu = torch.randn(batch_size, config.TEXT_EMBEDDING_DIM)
netG2 = GeneratorS2(config)
img2, mu2 = netG2(img1, cls_embedding, stage1_mu)
print(f"GeneratorS2 output: {img2.shape}")
print(f"Output corretto 215x215: {img2.shape[-2:] == (215, 215)}")
