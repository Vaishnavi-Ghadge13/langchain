import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env if using local environment variable
load_dotenv()

# Initialize Gemini with instructions
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,          # lower = more factual
    max_output_tokens=1024,   # longer answers
)

print("🤖 Gemini Chatbot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye 👋")
        break

    # Add instruction so Gemini always answers
    prompt = f"""You are a helpful AI assistant.
Answer the user's question in detail and never reply with just one word.

Question: {user_input}
Answer:"""

    try:
        response = chat.invoke(prompt)
        print("Bot:", response.content.strip())
    except Exception as e:
        print("⚠️ Error:", str(e))
