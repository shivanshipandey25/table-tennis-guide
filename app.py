import os
import uuid
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import numpy as np

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
AI21_API_KEY = os.getenv("AI21_API_KEY")

# Initialize MongoDB
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
chats_collection = db["chats"]
documents_collection = db["documents"]

# Initialize embeddings and AI21
embedder = SentenceTransformer("all-MiniLM-L6-v2")
ai21_client = AI21Client(api_key=AI21_API_KEY)

# Flask app
app = Flask(__name__)

def vector_search(query, search_mode="hybrid", top_k=5):
    """Simple similarity search without $vectorSearch"""
    query_embedding = embedder.encode(query)
    
    # Get documents based on search mode
    if search_mode == "local":
        all_docs = list(documents_collection.find({"category": "table_tennis"}))
    elif search_mode == "global":
        all_docs = list(documents_collection.find())
    else:  # hybrid
        all_docs = list(documents_collection.find())
    
    if not all_docs:
        return []
    
    similarities = []
    for doc in all_docs:
        if 'embedding' in doc:
            doc_embedding = np.array(doc['embedding'])
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc, similarity))
    
    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in similarities[:top_k]]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query") if request.json else None
    search_mode = request.json.get("search_mode", "hybrid")
    session_id = request.json.get("session_id", str(uuid.uuid4())) if request.json else str(uuid.uuid4())

    if not user_query or user_query.strip() == "":
        return jsonify({"error": "Query is required and cannot be empty"}), 400

    # Store user query
    chats_collection.insert_one({"session_id": session_id, "role": "user", "content": user_query})

    try:
        # Perform vector search
        search_results = vector_search(user_query, search_mode)
        
        # Build context from search results
        context = ""
        if search_results:
            context = "\n\n".join([doc.get("content", "") for doc in search_results[:3]])
        
        # Create RAG prompt
        rag_prompt = f"""Context: {context}

Question: {user_query}

Please answer the question based on the provided context. If the context doesn't contain relevant information, provide a general answer about table tennis."""

        messages = [ChatMessage(content=rag_prompt, role="user")]
        
        response = ai21_client.chat.completions.create(
            messages=messages,
            model="jamba-mini",
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
    except Exception as e:
        response_text = f"Error: {str(e)}"

    # Save AI response
    chats_collection.insert_one({"session_id": session_id, "role": "assistant", "content": response_text})

    return jsonify({
        "response": response_text, 
        "session_id": session_id,
        "search_mode": search_mode,
        "sources_found": len(search_results) if 'search_results' in locals() else 0
    })

if __name__ == "__main__":
    app.run(debug=True)