from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model='gpt-4o')

template = " write a {tone} email to {company} expresing interest in the {position} position, mentioning {skill} as a key strength."

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "formal",
    "company": "Google",
    "position": "Software Engineer",
    "skill": "Python"
})

result = llm.invoke(prompt)

print(result.content)