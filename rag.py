from typing import List
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()


def text_extract(pdf_path: str) -> str:
    "Extract text from a PDF file given its path."

    pdf_pages = []

    with open(pdf_path, 'rb') as file:

        pdf_reader = PdfReader(file)    

        for page in pdf_reader.pages:

            text = page.extract_text()
            pdf_pages.append(text)
    
    pdf_text = "\n".join(pdf_pages)

    return pdf_text


import requests

pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
response = requests.get(pdf_url)

pdf_path = 'attention_is_all_you_need.pdf'
with open(pdf_path, 'wb') as file:
    file.write(response.content)

pdf_text = text_extract(pdf_path)

from typing import List
import re
from collections import deque

def chunk_text(text: str, max_length=1000) -> List[str]:
    "Chunk text into smaller pieces without breaking sentences."
    
    sentences = deque(re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')))

    chunks = []
    chunk_text = ""

    while sentences:
        sentence = sentences.popleft().strip()
        if not sentence:
            continue

        if len(chunk_text) + len(sentence) + 1 <= max_length: 
            if chunk_text:
                chunk_text += " " + sentence
            else:
                chunk_text = sentence
        else:
            if chunk_text:
                chunks.append(chunk_text)
            chunk_text = sentence

    if chunk_text:
        chunks.append(chunk_text) 

    return chunks


chunks = chunk_text(pdf_text, max_length=1000)
##Create a vector store


import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models import Collection



def create_vector_store(db_path: str) -> Collection:
    "Create a persisten ChromaDB vector stroe with OpenAI embeddings."


    #Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    #Working with embeddings

    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
        )
    
    db = client.create_collection(
        name = "pdf_chunks",
        embedding_function=embeddings
    )

    return db

import os 
import uuid


def insert_chunks_vectordb(chunks: List[str], file_path: str) -> None:
    "Insert text chunks into the vector store with metadata."

    #Extract file name from path
    file_name = os.path.basename(file_path)

    id_list = [str(uuid.uuid4()) for _ in range(len(chunks))]

    #create metadata for each chunk

    metadata_list = [{"chunk": i, "source": file_name} for i in range(len(chunks))]
    
    #Define batch size for inserting chunks to optimize performance
    batch_size = 40

    #Inset chunks into database in batches

    for i in range(0, len(chunks), batch_size):
        end_id =  min(i + batch_size, len(chunks))
        db.add(
            documents=chunks[i:end_id],
            metadatas = metadata_list[i:end_id],
            ids = id_list[i:end_id]
        )
    print(f"{len(chunks)} chunks inserted into the vector store")
    
    #Retrieve chunks

from typing import List, Any

def retrieve_chunks(db: Collection, query:str, n_results: int =2) -> List[Any]:
    "Retrieve relevant chunks from the vector store based on a query."

    relevant_chunks = db.query(query_texts = [query], n_results=n_results)

    return relevant_chunks

#Build context:
def build_context(relevant_chunks) -> str:
    "Build a context string from the retrieved chunks for question answering."

    context = "\n".join(relevant_chunks["documents"][0])

    return context

#Build Pipeline
import os
from typing import Tuple


def get_context(pdf_path: str, query:str, db_path: str) -> Tuple[str, str]:
    "Retrieves the relevant chunks from the vector store and builds the context from them"

    #Check if the vectorstore already exists
    if os.path.exists(db_path):
        print("Loading existing vector store...")

        #initialize vector store
        client = chromadb.PersistentClient(path=db_path)

        #Create embedding function
        embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        #Get the collection of PDF chunks from the existing vector store
        db = client.get_collection(name="pdf_chunks", embedding_function=embeddings)
    else:
        print("Creating a new vector store..")

        pdf_text = text_extract(pdf_path)

        chunks = chunk_text(pdf_text)
        db = create_vector_store(db_path)
        insert_chunks_vectordb(chunks, db, pdf_path)

    relevant_chunks = retrieve_chunks(db, query)

    context = build_context(relevant_chunks)
    
    return context, query


def get_prompt(context: str, query: str) -> str:
    "Generates rag prompt based on the given context and query."

    rag_prompt = f"""You are an AI model trained for question answering. You should answer the given question based on the given context onlu.
    Question : {query}
    Context : {context}
    IF the answer is not present in the context, say "I don't know" or "The answer is not in the context"
    """
    return rag_prompt

from litellm import completion

def get_response(rag_prompt: str) -> str:
    """
    Sends a prompt to the OpenAI LLM and returns the answer.

    """
    model = "openai/gpt-4o-mini"
    
    messages = [{"role": "user", "content": rag_prompt}]

    response = completion(model=model, messages=messages, temperature=0)

    answer = response.choices[0].message.content

    cost = response._hidden_params.get("response_cost", 0.0)
    
    return answer, cost

answer, cost = get_response("Explain attention in simple terms")

print(answer)
print(f"Cost: ${cost:.8f}")

def rag_pipeline(pdf_path:str, query:str, db_path:str) ->str:
    """ Runs a RAG pipeline to answer a question based on the content of a PDF document."""

    #Get the context 

    context ,query = get_context(pdf_path, query, db_path)

    #Generate the rag prompt baed on context and query

    rag_prompt = get_prompt(context, query)
    #Get the response from the LLM
    response = get_response(rag_prompt)

    return response

#Run RAG pipeline

current_dir = "/content/rag"
persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")

pdf_path = "/home/sidd/Desktop/Python/Awajai/attention_is_all_you_need.pdf"

query = "Explain attention in simple terms"

answer = rag_pipeline(pdf_path, query, persistent_directory)