import json
import faiss
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)


#1. DATA LAYER
# Load the employee data from the JSON file
with open("employees.json", "r") as f:
    employees_data = json.load(f)

# 2. AI/ML COMPONENT (RAG SYSTEM) - RETRIEVAL SETUP
#Initialize the Sentence Transformer model to create embeddings
#This model converts text into dense vector representation
model = SentenceTransformer('all-MiniLM-L6-v2')


# Prepare the data for embedding
# We'll create a single text document for each employee by combining their details
employee_docs = [
    f"Name: {emp['name']}. Skills: {', '.join(emp['skills'])}. Experience: {emp['experience_years']} years. Projects: {', '.join(emp['projects'])}."
    for emp in employees_data["employees"]
]

#Create embeddings (vectors) for each employee document
print("Creating embeddings for employee data...")
employee_embeddings = model.encode(employee_docs)
print("Embeddings created.")

# Get the dimension of the embeddings (this is required by FAISS)
d = employee_embeddings.shape[1]

# Create a FAISS index. IndexFlatL2 is a simple index for a brute-force search.
index = faiss.IndexFlatL2(d)

# Add the embeddings to the index
# FAISS requires the vectors to be of type float32
index.add(np.array(employee_embeddings).astype('float32'))
print("FAISS index created with", index.ntotal, "vectors.")



# 3. BACKEND API (FASTAPI)
# Initialize the FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the HR Chatbot API!"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    # a. Retrieval: find relevant employees
    query_embedding = model.encode([request.query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=3)

    retrieved_employees = [employees_data["employees"][i] for i in indices[0]]

    context_text = ""
    for emp in retrieved_employees:
        context_text += (
            f"Name: {emp['name']}, Skills: {','.join(emp['skills'])}, "
            f"Experience: {emp['experience_years']} years, Projects: {','.join(emp['projects'])}.\n"
        )

    #b. Augmentation: Combine with a prompt
    system_prompt = (
        "You are an intelligent HR assistant. Your task is to recommend employees based on the provided "
        "query and employee data. Use only the given context to answer the questions. If the context does not contain "
        "the answer, state that you cannot help with the request. Be helpful and professional. Your response should "
        "be in natural language, not just raw data."
    )
    user_prompt = f"User query: {request.query}\n\nRelevant Employees:\n{context_text}"


    #c. Generation: Call the LLM
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {"response": response.choices[0].message.content}

# This endpoint is for a simple keyword search, without RAG
@app.get("/employees/search")
def search_employees(query: str):
    found_employees = []
    query_lower = query.lower()

    for emp in employees_data["employees"]:
        #Search for the query in skills, projects and name
        if (query_lower in emp['name'].lower() or 
            any(query_lower in skill.lower() for skill in emp['skills']) or 
            any(query_lower in proj.lower() for proj in emp['projects'])):
            found_employees.append(emp)

    return {"results": found_employees}