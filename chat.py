from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("teste.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3")

db = Chroma.from_documents(chunks, embeddings)

llm = OllamaLLM(model="llama3")

print("Chat iniciado. Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Pergunta: ")

    if pergunta.lower() == "sair":
        break

    resultados = db.similarity_search(pergunta, k=3)

    contexto = "\n".join([doc.page_content for doc in resultados])

    prompt = f"""
Use o contexto abaixo para responder a pergunta.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

    resposta = llm.invoke(prompt)

    print("\nResposta:", resposta, "\n")