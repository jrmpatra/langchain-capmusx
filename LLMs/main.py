import os, platform
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

llm = ChatOllama(temperature=0, model="gemma3:270m")
response = llm.invoke("Who is the currrent Chief Minister of Odisha?")

print(response.content)