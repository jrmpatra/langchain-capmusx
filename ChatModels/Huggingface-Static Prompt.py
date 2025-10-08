from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)


st.header("Huggingface Chat Model with Langchain")
user_input = st.text_input("Enter your query here")
if st.button("Submit"):
    response = model.invoke(user_input)
    st.write(response.content)





# print(response.content)