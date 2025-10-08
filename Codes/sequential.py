from langgraph.graph import StateGraph,START, END
from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)

#Create a State
class llmstate(TypedDict):

    question: str
    answer: str


def llm_qa(state: llmstate) -> llmstate:

    question = state['question']
    prompt = f'Answer following queston {question}'
    answer = model.invoke(prompt).content
    state['answer'] = answer
    return state


#Create a graph
graph = StateGraph(llmstate)

#Add Nodes
graph.add_node('llm_qa', llm_qa)

#Add Edge
graph.add_edge(START, 'llm_qa' )
graph.add_edge('llm_qa', END)

workflow = graph.compile()

initial_state = {'question': 'How far is moon from earth?'}
answer = workflow.invoke(initial_state)
print(answer)