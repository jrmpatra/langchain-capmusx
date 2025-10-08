from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
import time

load_dotenv()

#1. Prompt Definations
prompt1 = PromptTemplate(
          template = 'Summary about the book in 2 lines:  {topic}.',
          input_variables = ['topic']
)


#2. Model Definations
llm1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=2.0)

#3. Parsers
parser = StrOutputParser()

#4. Chain Definations
chain = prompt1 | model1 | parser


st.title("Document Loader - PDFLoader")
file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
if file:
    temp_file = 'upload_file.pdf'
    with open(temp_file, "wb") as f:
        f.write(file.getbuffer())
    st.success("File uploaded successfully!")
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    st.write(f"**Total pages loaded:** {len(docs)}")
    st.write("Let Me Go Through the content of document")
    time.sleep(2)
    st.markdown("## Here is the summary of first pages of document")
    st.write(chain.invoke({'topic': docs[0].page_content}))
    st.markdown("## Here is the summary of Second pages of document")
    st.write(chain.invoke({'topic': docs[1].page_content}))
    

