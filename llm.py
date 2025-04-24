import os
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

# ✅ Initialize latest Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Already latest

# ✅ Load new embedding model (e.g., intfloat/e5-large-v2)
embedding_model = SentenceTransformer("intfloat/e5-base")


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_chunks(chunks):
    # E5 model recommends using specific prefixes
    processed_chunks = [f"passage: {chunk}" for chunk in chunks]
    embeddings = embedding_model.encode(processed_chunks, convert_to_numpy=True)
    return embeddings

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, chunks, index, embeddings, k=3):
    # E5 model recommends prefixing queries with "query:"
    query_embedding = embedding_model.encode([f"query: {query}"], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, query):
    prompt = f"""Answer the question based on the below context:

Context:
{context}

Question: {query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

def rag_query(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    relevant_chunks = retrieve_relevant_chunks(query, chunks, index, embeddings)
    context = "\n".join(relevant_chunks)
    return generate_answer(context, query)
