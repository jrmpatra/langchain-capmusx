from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel

load_dotenv()

#1. Model Definations

llm1 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=1.100)

llm2 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
model2 = ChatHuggingFace(llm=llm2, temperature=1.100)

llm3 = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct", task="text-generation")
model3 = ChatHuggingFace(llm=llm3, temperature=1.100)


#Prompt Definations

prompt1 = PromptTemplate(
          template = 'Generate a short and simple notes about  {topic}.',
          input_variables = ['topic']
)   

prompt2 = PromptTemplate(
          template = 'Generate a quiz of 10 questions about  {topic}.',
          input_variables = ['topic']
)   

prompt3 = PromptTemplate(
          template = 'Merge the provided notes and quiz into a single comprehensive study guide about {notes} and {quizs}.',
          input_variables = ['notes', 'quizs']   
)   

parser = StrOutputParser()

#Chain Definations
parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | model1 | parser,
        'quizs' : prompt2 | model2 | parser
    }
)
merged_chain = prompt3 | model3 | parser
chain = parallel_chain | merged_chain

st.title("Parallel Chain with HuggingFace Chat Model")
user_input = st.text_input("Enter the topic name")
if st.button("Submit"):
    result = chain.invoke({'topic': user_input})
    st.write(result)


chain.get_graph().print_ascii()