import json
import faiss
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
# The OpenAI import is no longer needed for the "Simple" approach
# from openai import OpenAI
# The dotenv import is no longer needed without the OpenAI key
# from dotenv import load_dotenv

# --- 1. CONFIGURATION AND IMPORTS ---
# No LLM client configuration is needed for the "Simple" method.

# --- 2. DATA LAYER & RAG RETRIEVAL SETUP ---
# This block runs once when the server starts, setting up the searchable index

# Load the employee data from the JSON file
try:
    with open("employees.json", "r") as f:
        # Load the entire JSON object, which is a dictionary with a key "employees"
        raw_data = json.load(f)
        # Extract the list of employees from the "employees" key
        employees_data = raw_data["employees"]
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading employee data: {e}. Please check your employees.json file.")
    employees_data = []

# Initialize the Sentence Transformer model to create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2') 

# Prepare the data for embedding by creating a single text document per employee
employee_docs = [
    f"Name: {emp['name']}. Skills: {', '.join(emp['skills'])}. Experience: {emp['experience_years']} years. Projects: {', '.join(emp['projects'])}."
    for emp in employees_data
]

# Create embeddings (vectors) for each employee document
print("Creating embeddings for employee data...")
employee_embeddings = model.encode(employee_docs)
print("Embeddings created.")

# Build the FAISS vector index for fast similarity search
d = employee_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(employee_embeddings).astype('float32'))
print("FAISS index created with", index.ntotal, "vectors.")

# --- 3. FASTAPI APPLICATION ENDPOINTS ---
app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the HR Chatbot API!"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    """
    Handles a user's query using a simplified RAG system with smart formatting.
    """
    try:
        # a. Retrieval: Find relevant employees using the FAISS index
        query_embedding = model.encode([request.query])
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k=3)
        
        retrieved_employees = [employees_data[i] for i in indices[0]]
        
        # b. Generation (Simple Formatting): Create a response string from the retrieved data
        if not retrieved_employees:
            return {"response": "I couldn't find any employees matching that query."}

        response_text = "Based on your query, here are the most relevant candidates:\n\n"
        for emp in retrieved_employees:
            response_text += (
                f"- **Name:** {emp['name']}\n"
                f"  - Skills: {', '.join(emp['skills'])}\n"
                f"  - Experience: {emp['experience_years']} years\n"
                f"  - Projects: {', '.join(emp['projects'])}\n\n"
            )
        
        return {"response": response_text}
    
    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/search")
def search_employees(query: str):
    """
    A simple search endpoint for employees based on keywords in their skills, projects, and name.
    """
    found_employees = []
    query_lower = query.lower()

    for emp in employees_data:
        if (query_lower in emp['name'].lower() or
            any(query_lower in skill.lower() for skill in emp['skills']) or
            any(query_lower in proj.lower() for proj in emp['projects'])):
            found_employees.append(emp)
    
    return {"results": found_employees}