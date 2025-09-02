# app.py
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Local RAG Chat", page_icon="🤖", layout="centered")
st.title("🤖 Local RAG Chatbot")

# 1️⃣ Load vector DB
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return vectordb.as_retriever()

retriever = load_vectorstore()

# 2️⃣ Load local LLM
@st.cache_resource
def load_llm():
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=150
    )
    return HuggingFacePipeline(pipeline=generator)

llm = load_llm()

# 3️⃣ Build retrieval QA chain
@st.cache_resource
def load_qa():
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa = load_qa()

# 4️⃣ User input & display
question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    with st.spinner("🤖 Generating answer..."):
        answer = qa.run(question)
    st.markdown("**Answer:**")
    st.write(answer)
