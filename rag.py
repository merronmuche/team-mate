import os
import fitz  # PyMuPDF
import weaviate
import uuid
from typing import List
from dotenv import load_dotenv
from weaviate.weaviate_client import WeaviateClient


weaviate_interface = weaviate.setup_weaviate_interface()


load_dotenv()

# Weaviate client configuration
client = weaviate_interface.client

# Function to read and extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to chunk text into smaller parts
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to add documents and chunks to Weaviate
def add_document_chunks_to_weaviate(pdf_path: str):
    document_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(document_text)
    document_id = str(uuid.uuid4())

    # Create Document object
    client.create_object(
        data={"title": os.path.basename(pdf_path), "content": document_text, "wordCount": len(document_text.split()), "url": pdf_path},
        class_name="Document",
        # uuid=document_id
    )

    # Create DocumentChunk objects
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        client.create_object(
            data={"document": {"beacon": f"weaviate://localhost/Document/{document_id}"}, "text": chunk, "doc_name": os.path.basename(pdf_path)},
            class_name="DocumentChunk",
            # uuid=chunk_id
        )

# Directory containing PDF files
pdf_directory = "/home/meron/Documents/work/tenacious/team-mate/data"

# Loop through all PDFs in the directory and process them
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        add_document_chunks_to_weaviate(pdf_path)
        print(f"Processed {filename}")
        

print("All documents have been processed and added to Weaviate.")

