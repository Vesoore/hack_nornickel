import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device,  # or "mps" if on Apple Silicon
    ).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

if __name__ == "__main__":
    pass