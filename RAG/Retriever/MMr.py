from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
import time
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


# Create 10 simple Document objects
docs = [
    Document(page_content="Artificial Intelligence is transforming industries through automation and data-driven decisions.",
             metadata={"source": "AI Overview", "topic": "AI"}),
    
    Document(page_content="Machine learning allows systems to learn from data without being explicitly programmed.",
             metadata={"source": "ML Basics", "topic": "Machine Learning"}),
    
    Document(page_content="Deep learning uses neural networks with many layers to extract high-level features from data.",
             metadata={"source": "DL Concepts", "topic": "Deep Learning"}),
    
    Document(page_content="Natural language processing enables computers to understand and generate human language.",
             metadata={"source": "NLP Guide", "topic": "NLP"}),
    
    Document(page_content="Computer vision focuses on enabling machines to interpret and understand visual information.",
             metadata={"source": "CV Intro", "topic": "Computer Vision"}),
    
    Document(page_content="Reinforcement learning trains agents to make decisions through trial and error.",
             metadata={"source": "RL Concepts", "topic": "Reinforcement Learning"}),
    
    Document(page_content="Generative AI models can create text, images, and music by learning patterns from large datasets.",
             metadata={"source": "GenAI Article", "topic": "Generative AI"}),
    
    Document(page_content="Ethics in AI involves fairness, accountability, transparency, and privacy concerns.",
             metadata={"source": "AI Ethics", "topic": "Ethics"}),
    
    Document(page_content="RAG (Retrieval-Augmented Generation) combines document retrieval with generative models for better factual accuracy.",
             metadata={"source": "RAG Overview", "topic": "RAG"}),
    
    Document(page_content="Vector databases store embeddings to enable efficient similarity searches for large-scale AI applications.",
             metadata={"source": "VectorDB Basics", "topic": "Vector Databases"})
]

#1. Initializa Embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#2. Vector Store Database
vector_store = FAISS.from_documents(
        docuements=docs,
        embedding = model

)

#3. Enable MMR as retriever

retriever = vector_store.as_retriever(
        search_type='mmr',      #Enables MMR
        search_kwargs={"k":3, "lambda_mult": 1}   #lambda_mult - vary from 0 to 1 = relevance diversity balance

)

query = "what is Deep learning ?"
results = retriever.invoke(query)

print(results)