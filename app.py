import streamlit as st
from llm import rag_query
import tempfile
import os

st.set_page_config(page_title="PDF Q&A with Gemini", layout="centered")
st.title("📄 PDF Question Answering using  RAG")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the PDF content:")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing..."):
        answer = rag_query(tmp_path, query)
        st.success("Answer:")
        st.write(answer)

    os.remove(tmp_path)
