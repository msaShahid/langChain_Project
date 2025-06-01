from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}."),
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}."),
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
])

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
])

classification_chain = classification_template | model | StrOutputParser()

branches = RunnableBranch(
    (lambda x: x["sentiment"].lower().strip() == "positive", positive_feedback_template | model | StrOutputParser()),
    
    (lambda x: x["sentiment"].lower().strip() == "negative", negative_feedback_template | model | StrOutputParser()),
    
    (lambda x: x["sentiment"].lower().strip() == "neutral", neutral_feedback_template | model | StrOutputParser()),

    escalate_feedback_template | model | StrOutputParser()
)

chain = (
    RunnableLambda(lambda x: {
        "feedback": x["feedback"],
        "sentiment": classification_chain.invoke({"feedback": x["feedback"]})
    }) 
    | branches
)

# Example feedback input
review = "The product is terrible. It broke after just one use and the quality is very poor."

result = chain.invoke({"feedback": review})

print(result)
