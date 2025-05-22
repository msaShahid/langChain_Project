from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o')

messages = [
    SystemMessage("Hello, how can I assist you today?"),
    HumanMessage("I'm looking for a new phone. Can you recommend some options?"),
    AIMessage("I'd be happy to help you with that! What's your budget for the new phone?"),
    HumanMessage("I'm looking to spend around $500-$700"),
]

result = llm.invoke(messages)

print(result.content)  