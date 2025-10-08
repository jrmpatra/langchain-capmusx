from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)


###Indexing
### 1. 
video_id = "MdeQMVBuGgY" # only the ID, not full URL

transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
transcript = " ".join(chunk["text"] for chunk in transcript_list)
print(transcript)
