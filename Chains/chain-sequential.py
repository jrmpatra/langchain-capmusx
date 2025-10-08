from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


load_dotenv()

prompt1 = PromptTemplate(
          template = 'generate a detailed report on {topic}.',
          input_variables = ['topic']
)

promtp2 = PromptTemplate(
    template = 'Summerize the report about {topic} in bullet points in 10 lines.',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)
parser = StrOutputParser()

chain = prompt1 | model | parser | promtp2 | model | parser

st.title("Sequential Chain with HuggingFace Chat Model")
user_input = st.text_input("Enter the topic name")

if st.button("Submit"):
    result = chain.invoke({'topic': {user_input}})
    st.write(result)

