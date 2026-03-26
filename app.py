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

# ===== MONGODB SETUP =====
# Match exactly what your ingestion script uses
client = MongoClient(MONGO_URI)
db = client["table_tennis_bot"]
chats_collection = db["chat_history"]
documents_collection = db["chatbot_chunks"]  # fixed: was "documents"

# ===== MODELS =====
embedder = SentenceTransformer("all-MiniLM-L6-v2")
ai21_client = AI21Client(api_key=AI21_API_KEY)

# Flask app
app = Flask(__name__)


def vector_search(query, top_k=5):
    """Cosine similarity search over stored embeddings"""
    query_embedding = embedder.encode(query)

    all_docs = list(documents_collection.find({}, {"text": 1, "source": 1, "embedding": 1}))

    if not all_docs:
        return []

    similarities = []
    for doc in all_docs:
        if "embedding" in doc:
            doc_embedding = np.array(doc["embedding"])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc, float(similarity)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in similarities[:top_k]]


def get_chat_history(session_id, limit=6):
    """Fetch last N messages for this session to maintain conversation context"""
    history = list(
        chats_collection.find(
            {"session_id": session_id},
            {"role": 1, "content": 1}
        ).sort("_id", -1).limit(limit)
    )
    history.reverse()  # oldest first
    return history


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_query = data.get("query", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not user_query:
        return jsonify({"error": "Query is required and cannot be empty"}), 400

    # Save user message
    chats_collection.insert_one({
        "session_id": session_id,
        "role": "user",
        "content": user_query
    })

    search_results = []
    response_text = ""

    try:
        # Step 1: Vector search
        search_results = vector_search(user_query, top_k=5)

        # Step 2: Build context from top chunks
        if search_results:
            context_parts = []
            for doc, score in search_results[:3]:
                source = doc.get("source", "unknown")
                text = doc.get("text", "")   # fixed: was "content"
                context_parts.append(f"[Source: {source}]\n{text}")
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant knowledge base content found."

        # Step 3: Build conversation history for multi-turn memory
        history = get_chat_history(session_id, limit=6)
        messages = []

        # System message
        messages.append(ChatMessage(
            role="system",
            content=(
                "You are a Table Tennis expert assistant. "
                "Answer questions using the provided context from the knowledge base. "
                "If the context is not relevant, use your general table tennis knowledge. "
                "Be concise, accurate, and helpful."
            )
        ))

        # Add previous turns for memory
        for turn in history[:-1]:  # exclude current user message (added below)
            messages.append(ChatMessage(
                role=turn["role"],
                content=turn["content"]
            ))

        # Final user message with context injected
        rag_prompt = f"""Context from knowledge base:
{context}

User Question: {user_query}

Answer based on the context above. If it's not relevant, answer from general table tennis knowledge."""

        messages.append(ChatMessage(role="user", content=rag_prompt))

        # Step 4: Call AI21 Jamba
        response = ai21_client.chat.completions.create(
            messages=messages,
            model="jamba-mini",
            max_tokens=1000
        )

        response_text = response.choices[0].message.content

    except Exception as e:
        response_text = f"Error generating response: {str(e)}"

    # Save assistant response
    chats_collection.insert_one({
        "session_id": session_id,
        "role": "assistant",
        "content": response_text
    })

    return jsonify({
        "response": response_text,
        "session_id": session_id,
        "sources_found": len(search_results),
        "top_sources": [
            {"source": doc.get("source", "unknown"), "score": round(score, 4)}
            for doc, score in search_results[:3]
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)