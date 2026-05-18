from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os 
import pandas as pd
from langchain.vectorstores import Chroma
df = pd.read_csv("C:\\Users\\Parth Desai\\Desktop\\AIAGENTWITHPYTHON\\realistic_restaurant_reviews.csv")
emneddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i ,rows in df.iterrows():
        document = Document(
            page_content=rows["Title"]+""+rows["Review"],
            metadata ={"rating": rows["Rating"],"date":rows["Date"]},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)
vector_store = Chroma(
    collection_name = "restaurent_reviews",
    persist_directory=db_location,
    embedding_function=embeddings


)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs= {"k":5}
)