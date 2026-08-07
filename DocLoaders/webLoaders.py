from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader, FireCrawlLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = FireCrawlLoader(
    url="https://leetcode.com/problems/smallest-divisible-digit-product-ii/description/",
    mode="scrape"
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

print(docs[0].page_content)