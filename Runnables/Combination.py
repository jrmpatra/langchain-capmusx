from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

#1. Prompt Template
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a Detailed Report on {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["report"],
    template="Summerize the report in 10 words: {report}",
)

prompt3 = PromptTemplate(
    input_variables=["report"],
    template="Provide an explanation for this: {report}",
)
#2. Model Definations
llm1 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model1 = ChatHuggingFace(llm=llm1, temperature=1.100)


llm2 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
model2 = ChatHuggingFace(llm=llm1, temperature=1.100)

#3. Parsers
parser = StrOutputParser()

#4. Custom Definations

def word_count(text: str):
    return len(text.split())


#4. Chain Definations
chain = RunnableSequence(prompt1,model1,parser)


parallel_chain = RunnableParallel({
    "joke": RunnableBranch(
        (lambda x: len(x.split()) > 500, RunnableSequence(prompt2 , model2 , parser)),
        RunnablePassthrough()
    ),
    "explanation": RunnableSequence(prompt3 , model2 , parser),
    "countOfWords": RunnableLambda(word_count)
})

merged_chain = RunnableSequence(chain, parallel_chain)

results = merged_chain.invoke({'topic': 'Artificial Intelligence'})
print(results['joke'])