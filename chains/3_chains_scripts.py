from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(model="gtp-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("System", "You are facts export who knows facts about {animal}"),
        ("Human", "Tell me {fact_count} facts."),
    ]
)

