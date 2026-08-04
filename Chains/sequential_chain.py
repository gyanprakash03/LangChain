from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b')

prompt1 = PromptTemplate(
    template="Give me the names of any 5 countries in the continent of {continent}",
    input_variables=["continent"]
)

prompt2 = PromptTemplate(
    template="Name the country among these that is the largest in terms of area: \n {countries}",
    input_variables=["countries"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
 
result = chain.invoke({"continent": "Europe"})

print(result)