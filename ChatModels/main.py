import os, platform
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(temperature=0.100, model="llama3.2:3b")
response = model.invoke("Tell me a joke on IT")

print(response.content)