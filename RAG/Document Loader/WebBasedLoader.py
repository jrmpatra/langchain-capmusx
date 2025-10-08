from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#1. Model Defination
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=2.0)


#2. Prompt Definations
anti_hallucination_prompt = PromptTemplate(
    input_variables=[ "question"],
    template=(
        "You are a careful and factual AI assistant.\n"
        "Use only the information provided in the <context> section to answer the question.\n"
        "Question: {question}\n\n"
        "Instructions:\n"
        "1. Think step-by-step based only on the context.\n"
        "2. Do not use any external knowledge.\n"
        "3. Avoid speculation or assumptions.\n"
        "4. Keep your answer short, factual, and clear.\n\n"
        "Answer:"
    )
)

#3. Inputs
url = 'https://en.wikipedia.org/wiki/India'
loader = WebBaseLoader(url)
docs = loader.load()


#4. Parsers
parser = StrOutputParser()

#5. Chain Definations
chain = anti_hallucination_prompt | model | parser

print(chain.invoke({'question': "When India got independent?"}))


