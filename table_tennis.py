import os
import uuid
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from ai21 import AI21Client
from ai21.models.chat import ChatMessage

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
JAMBA_API_KEY = os.getenv("JAMBA_API_KEY")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
chunks_col = db["table_tennis_chunks"]
chatlog_col = db["chat_history"]

# Load model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Jamba
jamba = AI21Client(api_key=JAMBA_API_KEY)

# Separate conversation histories
local_conversation_history = []
global_conversation_history = []

def is_table_tennis_related(query, context_chunks):
    """
    Check if the query and retrieved context are actually table tennis related
    """
    table_tennis_keywords = [
        'table tennis', 'ping pong', 'paddle', 'bat', 'serve', 'rally', 'spin',
        'forehand', 'backhand', 'smash', 'loop', 'chop', 'block', 'push',
        'tournament', 'ittf', 'rubber', 'blade', 'net', 'table', 'ball',
        'point', 'game', 'set', 'match', 'doubles', 'singles', 'service',
        'return', 'topspin', 'backspin', 'sidespin', 'footwork', 'stance'
    ]
    
    # Check if query contains table tennis keywords
    query_lower = query.lower()
    query_has_tt_keywords = any(keyword in query_lower for keyword in table_tennis_keywords)
    
    # Check if context contains table tennis keywords
    context_text = " ".join(context_chunks).lower()
    context_has_tt_keywords = any(keyword in context_text for keyword in table_tennis_keywords)
    
    # Both query and context should be table tennis related
    return query_has_tt_keywords or context_has_tt_keywords

def get_local_answer(user_query):
    all_chunks = list(chunks_col.find({}))
    
    if not all_chunks:
        return None, None, "No documents found in local database"

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_model.encode(texts, convert_to_tensor=True)
    query_embedding = embed_model.encode(user_query, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    top_k = 3
    top_indices = similarity_scores.topk(k=top_k).indices
    top_chunks = [texts[i] for i in top_indices]
    top_scores = [float(similarity_scores[i]) for i in top_indices]

    highest_score = top_scores[0]
    
    # Increased similarity threshold for better relevance
    if highest_score < 0.3:  # Increased to 0.3
        return None, None, f"No relevant content found (highest similarity: {highest_score:.3f})"
    
    # Check if the content is actually table tennis related
    if not is_table_tennis_related(user_query, top_chunks):
        return None, None, "Query not related to table tennis content in documents"

    combined_context = "\n\n".join(top_chunks)
    return combined_context, highest_score, None

def ask_jamba_local(prompt):
    local_conversation_history.append(ChatMessage(role="user", content=prompt))
    
    try:
        response = jamba.chat.completions.create(
            messages=local_conversation_history,
            model="jamba-large",
            max_tokens=1024
        )
        
        assistant_response = response.choices[0].message.content.strip()
        local_conversation_history.append(ChatMessage(role="assistant", content=assistant_response))
        
        return assistant_response
    except Exception as e:
        return f"Jamba API error: {str(e)}"

def ask_jamba_global(user_input):
    global_conversation_history.append(ChatMessage(role="user", content=user_input))
    
    try:
        response = jamba.chat.completions.create(
            messages=global_conversation_history,
            model="jamba-large",
            max_tokens=1024
        )
        
        assistant_response = response.choices[0].message.content.strip()
        global_conversation_history.append(ChatMessage(role="assistant", content=assistant_response))
        
        return assistant_response
    except Exception as e:
        return f"❌ Jamba API error: {str(e)}"

def display_menu():
    print("\n" + "="*50)
    print("🏓 TABLE TENNIS CHATBOT - ANSWER OPTIONS")
    print("="*50)
    print("1. Local Answer (from your table tennis documents)")
    print("2. Global Answer (general AI knowledge)")
    print("3. Both Answers (local + global)")
    print("4. Exit")
    print("="*50)

def get_user_choice():
    while True:
        try:
            choice = input("Choose an option (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return 4

def save_to_chatlog(session_id, user_query, local_answer, global_answer, similarity_score=None, choice_made=None, local_error=None):
    chatlog_col.insert_one({
        "session_id": session_id,
        "timestamp": time.time(),
        "user_query": user_query,
        "local_answer": local_answer,
        "global_answer": global_answer,
        "similarity_score": similarity_score,
        "user_choice": choice_made,
        "local_error": local_error
    })

def process_local_answer(user_input):
    local_context, similarity, error_msg = get_local_answer(user_input)
    
    if error_msg:
        print(f"\nLOCAL ANSWER (from your table tennis documents):")
        print(f"{error_msg}")
        return None, None, error_msg
    
    if local_context:
        # Create a more restrictive prompt that emphasizes table tennis context
        local_prompt = f"""You are a table tennis expert assistant. Answer the question based ONLY on the provided table tennis context below. 

IMPORTANT RULES:
1. Only answer if the question is about table tennis
2. Only use information from the provided context
3. If the question is not about table tennis, respond with: "This question is not related to table tennis."
4. If the context doesn't contain relevant information, respond with: "I don't have information about this in my table tennis documents."

TABLE TENNIS CONTEXT:
{local_context}

QUESTION: {user_input}

ANSWER:"""
        
        local_response = ask_jamba_local(local_prompt)
        
        # Additional validation - check if the response indicates non-table tennis content
        if any(phrase in local_response.lower() for phrase in [
            "not related to table tennis",
            "don't have information about this",
            "not about table tennis",
            "outside my table tennis knowledge"
        ]):
            print(f"\nLOCAL ANSWER (from your table tennis documents):")
            print(f"Question not related to table tennis content in documents")
            return None, None, "Question not related to table tennis content"
        
        print(f"\nLOCAL ANSWER (from your table tennis documents):")
        print(f"Similarity Score: {similarity:.3f}")
        print(f"{local_response}")
        return local_response, similarity, None
    else:
        print(f"\nLOCAL ANSWER (from your table tennis documents):")
        print("No relevant table tennis content found in documents")
        return None, None, "No relevant content found"

def process_global_answer(user_input):
    global_response = ask_jamba_global(user_input)
    print(f"\nGLOBAL ANSWER (general AI knowledge):")
    print(f"{global_response}")
    return global_response

# Main Chatbot CLI with Options
print("🤖 Table Tennis Chatbot with Improved Local Validation")
print("Now properly validates table tennis content relevance!")
session_id = str(uuid.uuid4())

while True:
    print("\n" + "-"*60)
    user_input = input("🏓 You: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye! Thanks for using the Table Tennis Chatbot!")
        break
    
    if not user_input:
        print("Please enter a question.")
        continue
    
    # Display menu and get user choice
    display_menu()
    choice = get_user_choice()
    
    if choice == 4:  # Exit
        print("Goodbye! Thanks for using the Table Tennis Chatbot!")
        break
    
    local_response = None
    global_response = None
    similarity = None
    local_error = None
    
    print(f"\nProcessing your question: '{user_input}'")
    
    if choice == 1:  # Local only
        print("\nSearching your table tennis documents...")
        local_response, similarity, local_error = process_local_answer(user_input)
        choice_made = "local_only"
        
    elif choice == 2:  # Global only
        print("\nGetting answer from general AI knowledge...")
        global_response = process_global_answer(user_input)
        choice_made = "global_only"
        
    elif choice == 3:  # Both
        print("\nGetting both local and global answers...")
        
        # Get local answer
        local_response, similarity, local_error = process_local_answer(user_input)
        
        # Get global answer
        global_response = process_global_answer(user_input)
        
        choice_made = "both"
    
    # Save to chat log
    save_to_chatlog(
        session_id=session_id,
        user_query=user_input,
        local_answer=local_response,
        global_answer=global_response,
        similarity_score=similarity,
        choice_made=choice_made,
        local_error=local_error
    )
    
    print(f"\nResponse saved to chat history (Session: {session_id[:8]}...)")