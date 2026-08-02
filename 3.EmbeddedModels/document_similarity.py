from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=256)

documents = [
    "The capital of India is New Delhi.",
    "Python is a popular programming language for machine learning.",
    "LangChain helps developers build applications powered by large language models.",
    "The Eiffel Tower is located in Paris, France.",
    "Cricket is one of the most popular sports in India.",
]

query = "What is langchain?"

doc_vector = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

similarity = cosine_similarity([query_vector], doc_vector)[0]

sorted_similarity = sorted(enumerate(similarity), key=lambda x:x[1])
idx, score = sorted_similarity[-1]

print(query)
print(documents[idx])
print(f"similarity score: {score}")