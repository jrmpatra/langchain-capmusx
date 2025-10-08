from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough
from langchain_community.document_loaders import TextLoader

load_dotenv()

#1. Prompt Definations
prompt1 = PromptTemplate(
          template = 'Generate one liner tweet about  {topic}.',
          input_variables = ['topic']
)


#2. Model Definations
llm1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=2.0)

#3. Parsers
parser = StrOutputParser()

#4. Chain Definations
chain = prompt1 | model1 | parser

loader = TextLoader("file.txt",encoding="utf8")
docs = loader.load()


print(chain.invoke({'topic': docs[0].page_content}))

