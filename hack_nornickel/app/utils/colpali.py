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
    images = [
    Image.new("RGB", (224, 224), color="blue"),
    Image.new("RGB", (224, 224), color="black"),
    ]
    queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
    ]

# Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    print(scores)