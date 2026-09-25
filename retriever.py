import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("cbc-curriculum-rag")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
