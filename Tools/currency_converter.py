from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated


load_dotenv()


#1. Tool Defination

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """
  return base_currency_value * conversion_rate

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/54dd7424086d2349bc02a457/pair/{base_currency}/{target_currency}'
  response = requests.get(url)
  return response.json


result = get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
print(result)


convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})
print(convert)