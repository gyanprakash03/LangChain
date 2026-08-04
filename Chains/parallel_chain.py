from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b')

prompt1 = PromptTemplate(
    template="List the top 5 tourist attractions in {country}. Return only the attraction names as a numbered list.",
    input_variables=["country"]
)

prompt2 = PromptTemplate(
    template="List the top 5 traditional dishes of {country}. Return only the dish names as a numbered list.",
    input_variables=["country"]
)

prompt3 = PromptTemplate(
    template="""
        Using the following information about {country},

        Tourist Attractions:
        {attractions}

        Traditional Dishes:
        {dishes}

        Create a one-day travel itinerary that includes visiting some attractions and trying local food.
    """,
    input_variables=["country", "attractions", "dishes"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    country = RunnablePassthrough(),
    attractions = prompt1 | model | parser,
    dishes = prompt2 | model| parser
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"country" : "Mexico"})

print(result)