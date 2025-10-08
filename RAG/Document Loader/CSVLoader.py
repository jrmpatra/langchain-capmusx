from langchain_community.document_loaders import CSVLoader
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#1. Model Defination
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=0.3)


#2. Prompt Definations
anti_hallucination_prompt = PromptTemplate(
    input_variables=[ "question"],
    template=(
        "You are a careful and factual AI assistant.\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

#3. Inputs
loader = CSVLoader('file.csv')
docs = loader.load()


#4. Parsers
parser = StrOutputParser()

#5. Chain Definations
chain = anti_hallucination_prompt | model | parser

print(chain.invoke({'question': "What is the Age of Alice?"}))


