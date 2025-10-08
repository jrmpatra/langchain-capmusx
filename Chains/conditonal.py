from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

#1. Output Parser
parser = StrOutputParser()

#2. Model Definations
llm1 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=1.100)

#3.Prompt
prompt1 = PromptTemplate(
        template = 'Classifiy the following feedback into positive or negative. \n {feedback}',
        input_variables = ['feedback']
)

#First Chain
classfier_chain = prompt1 | model1 | parser


print(classfier_chain.invoke({'feedback': "The product quality is excellent but delivery was not prompt."}))