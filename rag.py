import asyncio
import os
import uuid
from typing import List
import fitz  # PyMuPDF
from dotenv import load_dotenv
import openai
import pandas as pd

import weaviate
from weaviate.weaviate_interface import WeaviateInterface

weaviate_interface = weaviate.setup_weaviate_interface()

# Load environment variables
load_dotenv()
client = weaviate_interface.client

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCHEMA_FILE = "/home/meron/Documents/work/tenacious/team-mate/weaviate/schema.json"
PDF_DIRECTORY = "/home/meron/Documents/work/tenacious/team-mate/data"


# Function to set up the Weaviate client interface asynchronously
async def setup_weaviate_client():
    """
    Asynchronous function to set up the Weaviate interface.
    """
    interface = WeaviateInterface(WEAVIATE_URL, OPENAI_API_KEY, SCHEMA_FILE)
    await interface.async_init()  # Initialize the interface asynchronously
    return interface.client


# Function to extract text from a PDF file asynchronously
async def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# Function to chunk text into smaller parts
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Chunks text into smaller parts.
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks


async def add_document_chunks_to_weaviate(client, pdf_path: str):
    """
    Adds a document and its chunks to Weaviate.

    Args:
        client (weaviate.Client): The Weaviate client object.
        pdf_path (str): The path to the PDF file.
    """
    document_text = await extract_text_from_pdf(pdf_path)
    chunks = chunk_text(document_text)
    # document_id = str(uuid.uuid4())

    # Create Document object using the Weaviate client
    document_id = await client.create_object(
        data={
            "title": os.path.basename(pdf_path),
            "content": document_text,
            "wordCount": len(document_text.split()),
            "url": pdf_path,
        },
        class_name="Document"
    )

    beacon = f"weaviate://localhost/Document/{document_id}"
    chunk_ids = []
    # Create DocumentChunk objects for each chunk
    for i, chunk in enumerate(chunks):
        # chunk_id = str(uuid.uuid4())
        chunk_id = await client.create_object(
            data={
                "document": [{"beacon": beacon}],
                "text": chunk,
                "doc_name": os.path.basename(pdf_path),
                # "embedding": chunk_embedding,
            },
            class_name="DocumentChunk"
        )
        
        chunk_ids.append(chunk_id)

    return document_id, chunk_ids


# The main function that processes all PDF files in the directory
async def main():
    """
    The main function that processes all PDF files in the directory.
    """
    client = await setup_weaviate_client()  # Get the Weaviate client

    processed_documents = []

    # Loop through all PDFs in the directory and process them
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            document_id, chunk_ids = await add_document_chunks_to_weaviate(
                client, pdf_path
            )
            processed_documents.append({
                "filename": filename,
                "document_id": document_id,
                "chunk_ids": chunk_ids
            })
            print(f"Processed {filename} with document ID {document_id}")

    # Display processed documents in a table
    df = pd.DataFrame(processed_documents)
    print(df)



if __name__ == "__main__":
    asyncio.run(main())
