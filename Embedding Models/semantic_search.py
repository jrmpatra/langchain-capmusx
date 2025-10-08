from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

load_dotenv()
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.",
    "LangChain is designed to be modular and flexible, allowing developers to easily integrate different components into their applications.",
    "The framework provides a variety of tools and utilities to help developers build and deploy their applications quickly and easily.",
    "LangChain is an open-source project and has a growing community of developers and users.",
    "It is actively maintained and updated with new features and improvements.",
    "LangChain is a powerful tool for anyone looking to build applications using language models.",
    "With its modular design and wide range of features, it makes it easy to create sophisticated applications that can understand and generate human-like text.",
    "Whether you're a seasoned developer or just getting started with language models, LangChain is definitely worth checking out.",
    "So, if you're interested in building applications that leverage the power of language models, be sure to give LangChain a try!",
]

query = "Langchain can be used for?"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embedding)[0]
index, scores = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[0]

print(query)
print(f"Most similar document (score: {scores}): {documents[index]}")