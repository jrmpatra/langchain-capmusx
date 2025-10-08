from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests

load_dotenv()

#1. Model Defination - for Reasoning Part
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)


# #3. Validate the LLM
# print(model.invoke('Hi'))

#2. Use of DuckDuckGoSearchRun tool and Waether Tool
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()


# #3. Validate the Tool
# print(search_tool.invoke('Top News of the day in one line'))


#Pull Agents and pull prompt from hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt


# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)


# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose=True
)


#Step 5 : Invoke Agent
response = agent_executor.invoke({"input":"suggest 3 ways to reach goa"})
print(response)