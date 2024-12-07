from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import torch
from app.utils.colpali import model, processor, device

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def preprocess_image(image):
    # Уменьшение размера изображения с помощью PIL
    img = image.resize((448, 448))
    return np.array(img)

def process_pdf(pdf_path):
    embs = []
    images = pdf_to_images(pdf_path)
    
    # Предобработка и отправка каждой страницы
    for i, image in enumerate(images):
        preprocessed_image = preprocess_image(image)
        
        # Отправка в ColPali
        batch_image = processor.process_images([Image.fromarray(preprocessed_image)])
        batch_image = {k: v.to(device) for k, v in batch_image.items()}
        print("start model")
        with torch.no_grad():
          image_embeddings = model(**batch_image)
        print("end model")
        
        embs.append({"page_num": i, "embedding": image_embeddings})
        
    return embs
