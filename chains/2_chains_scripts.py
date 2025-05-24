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

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_message()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"animal": "cat", "count" : 3})

print(response)