from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import streamlit as st


load_dotenv()


#1. Runnable 1 : Promprt Templates
prompt1 = PromptTemplate(
    template= 'Write a joke about {topic}.',
    input_variables=['topic']
)

#2. Runnable 2 : LLM Models
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.5)

#3. Runnable 3 : Output Parser
parser = StrOutputParser()

#6. Prompt2

prompt2 = PromptTemplate(
    template= 'Explain the joke {joke}.',
    input_variables=['joke']
)


#4. Chain with help of RunnableSequence
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

#5. Print the result
result = chain.invoke({'topic': 'dog'})
print(result)
