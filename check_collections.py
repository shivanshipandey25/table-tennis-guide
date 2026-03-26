from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://shivanshipandey:tabletennis123@chatbot-cluster.qjua5qb.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-cluster"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Connected successfully!\n")

    db_names_to_check = ["chatbot", "table_tennis_bot"]

    for db_name in db_names_to_check:
        print(f"\n📁 Database: {db_name}")
        db = client[db_name]
        collections = db.list_collection_names()

        if collections:
            for col in collections:
                count = db[col].count_documents({})
                print(f"   - Collection: {col} | Documents: {count}")
        else:
            print("   ❌ No collections found")

except Exception as e:
    print("❌ Error:", e)