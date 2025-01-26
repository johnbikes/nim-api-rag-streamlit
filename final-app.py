import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('us-census')
        st.session_state.docs = st.session_state.loader.load()
        print('st.session_state.docs', st.session_state.docs)
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        # top 30 docs?
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        print('st.session_state.final_documents', st.session_state.final_documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    print('done w vector_embedding')

def main():
    load_dotenv()

    nim_api_key = os.environ.get('NIM_API_KEY')
    print(f"nim_api_key = {nim_api_key}")

    os.environ['NVIDIA_API_KEY'] = nim_api_key

    llm = ChatNVIDIA(
        model="meta/llama-3.2-3b-instruct"
    )

    st.title('NIM API RAG Demo')

    vector_embedding()

    # prompt = ChatPromptTemplate.from_template("What is the {topic} of this article?")

if __name__ == "__main__":
    main()