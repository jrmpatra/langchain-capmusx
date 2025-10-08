from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2-Exp", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)

chat_history = []

while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break   
    else:
        response = model.invoke(chat_history)
        chat_history.append(AIMessage(content = response.content))
        print("Bot:", response.content)

print(chat_history)