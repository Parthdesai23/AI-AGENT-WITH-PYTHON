from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are expert in answering questions about a pizza restaurant

here are some relevant reviews: {reviews}
here is the question to answer{question}
"""
prompt  = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("_____________________________________________________")
    question = input("Ask question(q ro quit):")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":reviews,"question":question})
    print(result)
