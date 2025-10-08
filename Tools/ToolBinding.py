from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


#1. Tool Defination

search_tool = DuckDuckGoSearchRun()

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

##print(multiply.invoke({'a': 3, 'b': 10}))

#2. Model Definations
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)

#3. Tool Binding
agent = model.bind_tools([multiply])

#4. It suggests to call the tool. Storing the query and response of tools in messages now.
query = HumanMessage('How are you ?')
messages = [query]
result= agent.invoke(messages)
messages.append(result)

#5. Tool Calling
final_results = multiply.invoke(result.tool_calls[0])
messages.append(final_results)

#6. Tool Execution
print(agent.invoke(messages).content)