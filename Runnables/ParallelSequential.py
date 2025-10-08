from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import streamlit as st


load_dotenv()

#1. Prompt Definations
prompt1 = PromptTemplate(
          template = 'Generate one liner tweet about  {topic}.',
          input_variables = ['topic']
)

prompt2 = PromptTemplate(
          template = 'Generate a one liner Linkedin post about  {topic}.',
          input_variables = ['topic']
)

#2. Model Definations
llm1 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=1.100)

#3. Parsers
parser = StrOutputParser()

#4. Chain Definations
chain = RunnableParallel(
    {
        'tweet' : prompt1 | model1 | parser,
        'linkedin_post' : prompt2 | model1 | parser
    }

)

print(chain.invoke({'topic': 'Artificial Intelligence'}))