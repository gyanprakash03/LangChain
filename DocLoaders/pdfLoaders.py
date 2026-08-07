from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = PyMuPDFLoader(
    "AI Engineer Roadmap.pdf",
    extract_tables = "markdown",
)

docs = loader.load()

model = ChatGroq(model="openai/gpt-oss-20b")

# prompt = PromptTemplate(
#     template="""
#     Mention any 5 aptitude topics mentioned on this web page.

#     Webpage Content:
#     {content}
#     """,
#     input_variables=["content"]
# )

# parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({
#     "content": docs[0].page_content
# })

print(docs[1].page_content)