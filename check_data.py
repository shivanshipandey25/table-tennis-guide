from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGO_URI")

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