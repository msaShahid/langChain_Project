from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(model="gtp-4o")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("System", "You are facts export who knows facts about {animal}"),
        ("Human", "Tell me {fact_count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    ("System", "You are a translator and convert the provided text into {language}"),
    ("Human", "Translate the text: {text}"),
)

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())} \n {x}")

prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({"animal":"cat","count": 2})

print(result)