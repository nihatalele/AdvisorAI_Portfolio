import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_database():
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Initialize OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002"
    )

    # Load course data
    with open('cse_courses.json') as f:
        courses = json.load(f)

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name="courses")
        print("Deleted existing collection")
    except:
        print("No existing collection found")

    # Create new collection
    collection = chroma_client.create_collection(
        name="courses",
        embedding_function=openai_ef
    )

    # Prepare documents for embedding
    documents = []
    metadatas = []
    ids = []
    
    for i, course in enumerate(courses):
        # Create a comprehensive document for embedding
        doc = f"Course: {course['number']} - {course['title']}\n"
        doc += f"Description: {course['description']}\n"
        doc += f"Prerequisites: {course['prerequisites']}\n"
        doc += f"Units: {course['units']}"
        
        documents.append(doc)
        metadatas.append({
            "number": course['number'],
            "title": course['title'],
            "prerequisites": course['prerequisites'],
            "units": course['units']
        })
        ids.append(str(i))
    
    # Add documents to collection
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(documents)} documents to the database")
        
        # Verify the documents were added
        results = collection.query(
            query_texts=["database systems"],
            n_results=1
        )
        print("\nVerification query result:")
        print(results)
        
    except Exception as e:
        print(f"Error adding documents to collection: {e}")

if __name__ == "__main__":
    init_database() 