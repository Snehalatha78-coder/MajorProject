import os
import io
import faiss
import numpy as np
import pandas as pd
from langchain.document_loaders import UnstructuredPDFLoader
from sentence_transformers import SentenceTransformer
import fitz
import streamlit as st
from io import BytesIO
from gtts import gTTS
from audiorecorder import audiorecorder
import tempfile
import requests

# Helper Functions
def load_pdf(uploaded_file):
    """Extracts text and metadata from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    metadata = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
        metadata.append({"page_number": page_num + 1, "page_text": page.get_text()})
    return text, metadata

def load_excel(file_path):
    """Extracts text from an Excel file."""
    df = pd.read_excel(file_path)
    return "\n".join([" ".join(map(str, row)) for row in df.values])

def split_into_chunks(text, max_length=512):
    """Splits large text into smaller chunks for embedding."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return list(set(chunks))  # Remove duplicate chunks

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Generates embeddings for a list of texts."""
    embeddings = model.encode(texts)
    return np.array(embeddings)

# Initialize FAISS Index
embedding_dim = 384  # Match the model's dimensionality
index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity for FAISS
metadata_store = []
text_store = []

def store_in_faiss(embeddings, metadata, texts):
    """Stores embeddings, metadata, and actual texts in FAISS."""
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Embedding dimensionality mismatch: {embeddings.shape[1]} vs {embedding_dim}")
    index.add(embeddings)
    metadata_store.extend(metadata)
    text_store.extend(texts)

def retrieve(query, top_k=5):
    """Retrieves top-k similar documents for a given query."""
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    
    unique_results = set()
    results = []
    
    for idx, i in enumerate(indices[0]):
        if text_store[i] not in unique_results:
            unique_results.add(text_store[i])
            results.append((text_store[i], metadata_store[i], distances[0][idx]))
    
    return results

# Text-to-Speech Setup
def text_to_speech(text, lang='en'):
    """Converts text to speech and saves it to a temporary file."""
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            return temp_audio_file.name  # Return the path to the temporary file
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Generate response with an external API (example: Mistral API)
def generate_response(transcription):
    MISTRAL_API_KEY = ""  # Set your API key here
    MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"  # Replace with the actual Mistral endpoint
    
    if not MISTRAL_API_KEY:
        return "MISTRAL_API_KEY is not set."

    try:
        payload = {
            "model": "codestral-latest",
            "prompt": f"You are a helpful assistant. Summarize your response in 80-100 words.\nUser: {transcription}\nAI:",
            "max_tokens": 150,
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                return "The API response format is invalid or missing 'choices'."
        else:
            return f"Error {response.status_code}: {response.content.decode('utf-8', errors='ignore')}"
    except Exception as e:
        st.error(f"Error communicating with Mistral API: {e}")
        return "Unable to generate a response."

# Streamlit UI
st.title("Document Search and Retrieval with AI Assistant")
st.write("Upload a PDF or Excel file and enter a query to retrieve similar documents.")

# File Upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "xlsx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.write("Processing PDF...")
        document_text, metadata = load_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        st.write("Processing Excel...")
        document_text = load_excel(uploaded_file)
        metadata = [{} for _ in range(len(document_text))]

    chunks = split_into_chunks(document_text)
    st.write(f"Document split into {len(chunks)} chunks.")
    embeddings = embed_texts(chunks)
    store_in_faiss(embeddings, metadata, chunks)
    st.write("Embeddings and metadata stored in FAISS.")

    st.write("Speak your query (click the microphone icon):")
    audio_bytes = audiorecorder("Click to record", "Recording...")
    if audio_bytes:
        query = "sample query text"  # Replace with transcription logic
        results = retrieve(query)
        relevant_text = " ".join([result[0] for result in results])
        response_text = generate_response(relevant_text)
        st.markdown("### AI Assistant:")
        st.markdown(f"##### {response_text}")

        # Convert AI response to speech
        audio_file_path = text_to_speech(response_text)
        if audio_file_path:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            os.remove(audio_file_path)
