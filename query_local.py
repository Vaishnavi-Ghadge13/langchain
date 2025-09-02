# query_local.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# load vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever()

# local text generation model (free, CPU-friendly)
generator = pipeline(
    "text-generation",
    model="distilgpt2",  # small and free
    max_new_tokens=150
)
llm = HuggingFacePipeline(pipeline=generator)

# retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

print("🤖 Ask questions (type 'exit' to quit)")
while True:
    q = input("\nQuestion: ")
    if q.lower() == "exit":
        break
    ans = qa.run(q)
    print("\nAnswer:", ans)
