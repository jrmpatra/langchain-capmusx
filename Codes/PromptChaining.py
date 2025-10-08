from langgraph.graph import StateGraph,START, END
from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)




#Create a State
class BlogState(TypedDict):

    topic: str
    outline: str
    blog: str


#create_outline

def create_outline(state: BlogState) -> BlogState:

    topic = state['topic']
    prompt = f'Create an Outline in 1 line about {topic}'
    answer = model.invoke(prompt).content
    state['outline'] = answer
    return state

#create_blog
def create_blog(state: BlogState) -> BlogState:

    topic = state['topic']
    outline = state['outline']
    prompt = f'Create an Blog in 3 lines about {topic} using {outline}'
    answer = model.invoke(prompt).content
    state['blog'] = answer
    return state

#Create a graph
graph = StateGraph(BlogState)

#Add Nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)


#Add Edge
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

#Compile
workflow = graph.compile()

#Invoke
initial_state = {'topic': 'Virat Kohli'}
answer = workflow.invoke(initial_state)
print(answer)