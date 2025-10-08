from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough

load_dotenv()

print(RunnablePassthrough().invoke("test"))