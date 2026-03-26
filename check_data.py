from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://shivanshipandey:tabletennis123@chatbot-cluster.qjua5qb.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-cluster"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Connected successfully!\n")

    # Show all databases
    db_names = client.list_database_names()
    print("📁 Databases found:")
    for db in db_names:
        print("-", db)

except Exception as e:
    print("❌ Connection failed:", e)