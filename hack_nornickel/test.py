from app.utils.preprocess_pdf import process_pdf

pdf_path = "data/Лабораторные работы.pdf"
print("Processing PDF...")
embs = process_pdf(pdf_path)
print("PDF processed.")
print(embs)