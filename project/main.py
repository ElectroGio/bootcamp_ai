from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import psycopg2
import numpy as np
from openai import OpenAI
import ollama
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI()

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    model: str # "openai" or "ollama"

class ChatResponse(BaseModel):
    response: str

# Services
def get_embedding(text, model="openai"):
    if model == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    elif model == "ollama":
        response = ollama.embeddings(model='mistral', prompt=text)
        return response["embedding"]
    else:
        raise ValueError("Unsupported model")

def find_similar_chunks(embedding, conn):
    embedding_array = np.array(embedding)
    cur = conn.cursor()
    cur.execute('SELECT "Content", "Embedding" FROM "RagDocumentChunks"')
    chunks = cur.fetchall()
    
    best_match = None
    highest_similarity = -1
    
    for content, db_embedding in chunks:
        if isinstance(db_embedding, str):
            db_embedding = json.loads(db_embedding)

        db_embedding_array = np.array(db_embedding)
        similarity = np.dot(embedding_array, db_embedding_array) / (np.linalg.norm(embedding_array) * np.linalg.norm(db_embedding_array))
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = content
            
    cur.close()
    return best_match

def get_chat_response(prompt, context, model="openai"):
    if model == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    elif model == "ollama":
        response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ]
        )
        return response['message']['content']
    else:
        raise ValueError("Unsupported model")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        conn = get_db_connection()
        
        # 1. Get embedding for the user's message
        embedding = get_embedding(request.message, request.model)
        
        # 2. Find the most similar document chunk
        similar_chunk = find_similar_chunks(embedding, conn)
        
        if not similar_chunk:
            return ChatResponse(response="I couldn't find any relevant information.")
            
        # 3. Generate a response using the selected model
        response_text = get_chat_response(request.message, similar_chunk, request.model)
        
        conn.close()
        
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
