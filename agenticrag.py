"""
Agentic RAG using LangGraph + ChromaDB + OpenAI GPT-3.5
Updated to support LangChain v0.2.x structure
"""

# STEP 1: Install dependencies
# !pip install -U langchain-openai langchain-community langgraph chromadb pymupdf

import os
import zipfile
from typing import TypedDict, List
import traceback

# STEP 2: Extract docs.zip
zip_path = 'docs.zip'
extract_to = './'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Files extracted to: {os.path.abspath(extract_to)}")

# STEP 3: Load documents
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

folder_path = "./docs"

loaders = [
    DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader),
    DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader),
    DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
    DirectoryLoader(folder_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
]

all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())

print(f"✅ Loaded {len(all_docs)} documents.")

# STEP 4: Split and embed documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

if not docs:
    raise ValueError("❌ No documents found for embeddiMarkdown Preview Mermaid Supportng.")

print("-------------------")
openai_api_key = "sk-proj-OYrlZFcR0kFJqmf55Fjoyycg1bTezoooXicKlZUJkc..............."
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

try:
    print("🔄 Embedding and creating vector store...")
    persist_path = os.path.abspath("./chroma_db")
    vectorstore = Chroma.from_documents(docs, embedding, persist_directory=persist_path)
    retriever = vectorstore.as_retriever()
    print("✅ Vector store ready.")
except Exception as e:
    print("❌ Exception during vector store creation:")
    traceback.print_exc()
    raise

print("✅ Vector store created and retriever ready.")

# STEP 5: Build LangGraph workflow
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, Document
from langchain.schema.messages import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

class ArchitectState(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_context(state: ArchitectState) -> ArchitectState:
    question = state["question"]
    results = retriever.invoke(question)
    print(f"🔍 Retrieved {len(results)} context documents.")
    return {**state, "context": results}

def architect_response(state: ArchitectState) -> ArchitectState:
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])
    system_prompt = (
        "You are a senior Software Architect. Use the provided technical knowledge base to answer with precision."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{context_text}")
    ]

    print("🧠 Calling LLM...")
    response = llm.invoke(messages)
    print("✅ LLM responded.")
    return {**state, "answer": response.content}

# STEP 6: Build the graph
workflow = StateGraph(ArchitectState)
workflow.add_node("RetrieveContext", retrieve_context)
workflow.add_node("ArchitectResponse", architect_response)
workflow.set_entry_point("RetrieveContext")
workflow.add_edge("RetrieveContext", "ArchitectResponse")
workflow.add_edge("ArchitectResponse", END)

agent_graph = workflow.compile()

# STEP 7: Run the agent
query = "Binary Trees—Must Know Algorithms"
initial_state: ArchitectState = {
    "question": query,
    "context": [],
    "answer": ""
}

try:
    final_state = agent_graph.invoke(initial_state)
    print("\n✅ Answer:\n")
    print(final_state["answer"])
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🏁 Script completed.")
