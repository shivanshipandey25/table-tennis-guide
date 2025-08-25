from flask import Flask, request, jsonify, render_template
import os
import uuid
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

load_dotenv()

app = Flask(__name__)

# Your existing DB, embeddings, AI21 setup here...

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    # your chatbot logic here...
    response_text = "Example reply: " + user_message  

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
