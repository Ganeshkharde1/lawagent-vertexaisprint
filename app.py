import os
import streamlit as st
import google.generativeai as genai
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Configure Gemini API
genai.configure(api_key="your-gemini-api-key")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="lawagent_db")

# Create or get a collection for legal documents
collection = chroma_client.get_or_create_collection(name="legal_docs")

# Load SentenceTransformer model for embedding generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to store extracted text in ChromaDB
def store_pdf_to_chromadb(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Split large text into smaller chunks
    chunk_size = 500  # Adjust based on legal text structure
    chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]

    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": pdf_path, "chunk_id": idx}],
            ids=[f"doc_{idx}"]
        )
    return f"✅ Successfully stored {len(chunks)} chunks from {pdf_path} in ChromaDB!"

# Function to query ChromaDB
def query_chromadb(user_query, top_n=3):
    results = collection.query(query_texts=[user_query], n_results=top_n)
    matched_laws = results['documents'][0] if results['documents'] else ["No relevant legal information found."]
    return matched_laws

# Function to generate response with Gemini
def generate_legal_response(matched_laws):
    prompt = f"Summarize the following legal provisions in simple terms:\n{matched_laws}"
    response = genai.generate_content(prompt=prompt, model="gemini-2")
    return response['content'] if 'content' in response else "Sorry, I couldn't generate a response."

# -------------- Streamlit UI --------------

st.title("⚖️ LawAgent - AI Legal Assistant")
st.markdown("🚀 **An AI-powered legal assistant using Gemini & ChromaDB**")

# File Upload Section
st.sidebar.header("📂 Upload Legal Documents (PDF)")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_path = f"./uploaded_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    status = store_pdf_to_chromadb(pdf_path)
    st.sidebar.success(status)

# Chatbot Section
st.subheader("🤖 Ask a Legal Question")
user_query = st.text_input("Enter your legal question:")

if st.button("Get Answer"):
    if user_query:
        matched_laws = query_chromadb(user_query)
        ai_response = generate_legal_response(matched_laws)
        
        st.write("### 📜 Legal Insight")
        st.write(f"**Relevant Law Sections:** {matched_laws}")
        st.write("**AI Explanation:**")
        st.write(ai_response)
    else:
        st.warning("⚠️ Please enter a legal question.")

# Footer
st.markdown("---")
st.markdown("🔹 **Developed using Gemini 2.0, ChromaDB & Streamlit** | 🚀 **Built for legal research & assistance**")
