import torch
import transformers
import torchvision

# Check versions
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)

# Check if PyTorch can see your GPU (if you have one)
print("CUDA available:", torch.cuda.is_available())

exit()