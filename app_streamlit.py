# app_streamlit.py
import streamlit as st
from query import get_retriever
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("Simple RAG Chat (LangChain + Chroma)")

if "chain" not in st.session_state:
    retriever = get_retriever()
    if "OPENAI_API_KEY" in st.secrets:
        llm = OpenAI(temperature=0)
    else:
        llm = OpenAI(temperature=0)  # replace with local LLM if no key; placeholder
    st.session_state.chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = st.text_input("Ask something about the documents:")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        ans = st.session_state.chain.run(query)
    st.markdown("**Answer:**")
    st.write(ans)
