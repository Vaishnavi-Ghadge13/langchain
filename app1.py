import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini 2.5 Flash Chatbot")

user_input = st.text_input("Ask a question:")

if st.button("Ask"):
    if user_input:
        try:
            answer = chat.invoke(user_input).content
            st.markdown(f"**🤖 Bot:** {answer}")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
