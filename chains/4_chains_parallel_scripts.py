from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic."),
    ("human", "Provide a brief summary of the movie {movie_name}."),
])

summary_chain = summary_template | model | StrOutputParser()

def analyze_plot_prompt(summary):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
    ])
    return plot_template.format_prompt(plot=summary)

def analyze_character_prompt(summary):
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
    ])
    return character_template.format_prompt(characters=summary)


plot_analysis_chain = RunnableLambda(analyze_plot_prompt) | model | StrOutputParser()
character_analysis_chain = RunnableLambda(analyze_character_prompt) | model | StrOutputParser()

analysis_parallel = RunnableParallel({
    "plot_analysis": plot_analysis_chain,
    "character_analysis": character_analysis_chain,
})

combine_chain = RunnableLambda(
    lambda x: f"Plot Analysis:\n{x['plot_analysis']}\n\nCharacter Analysis:\n{x['character_analysis']}"
)

full_chain = summary_chain | analysis_parallel | combine_chain

result = full_chain.invoke({"movie_name": "Inception"})

print(result)
