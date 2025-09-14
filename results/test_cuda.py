import torch
import sys
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("CUDA is available: ", torch.cuda.is_available())