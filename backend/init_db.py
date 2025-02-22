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

    # Delete existing collections if they exist
    for collection_name in ["courses", "course_titles"]:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            print(f"No existing collection found: {collection_name}")

    # Create two collections: one for titles and one for full descriptions
    title_collection = chroma_client.create_collection(
        name="course_titles",
        embedding_function=openai_ef
    )
    
    full_collection = chroma_client.create_collection(
        name="courses",
        embedding_function=openai_ef
    )

    # Prepare documents for embedding
    title_documents = []
    full_documents = []
    metadatas = []
    ids = []
    
    for i, course in enumerate(courses):
        # Standardize course number format (remove space if present)
        course_num = course['number'].replace(' ', '')
        
        # Create title-focused document with repeated course number for emphasis
        title_doc = f"""Course Number: {course_num} {course_num} {course_num}
Course: {course_num} - {course['title']}
Search Terms: {course_num} CSE {course_num[-4:]} {course['title']}"""
        
        # Create comprehensive document with emphasized course number
        full_doc = f"""Course Number: {course_num} {course_num} {course_num}
Course Title: {course_num} - {course['title']}
Description: {course['description']}
Prerequisites: {course['prerequisites']}
Units: {course['units']}
Search Terms: {course_num} CSE {course_num[-4:]} {course['title']}"""
        
        title_documents.append(title_doc)
        full_documents.append(full_doc)
        
        # Enhanced metadata with standardized course number
        metadatas.append({
            "number": course_num,
            "number_raw": course_num[-4:],  # Just the numeric part
            "title": course['title'],
            "prerequisites": course['prerequisites'],
            "units": course['units']
        })
        ids.append(str(i))
    
    # Add documents to collections
    try:
        title_collection.add(
            documents=title_documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(title_documents)} documents to title collection")
        
        full_collection.add(
            documents=full_documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(full_documents)} documents to full collection")
        
    except Exception as e:
        print(f"Error adding documents to collections: {e}")

    # Add verification step
    try:
        # Verify a few known courses
        test_courses = ["CSE3901", "CSE3902", "CSE3241"]
        print("\nVerifying course entries:")
        for test_course in test_courses:
            result = full_collection.get(
                where={"number": test_course}
            )
            if result['documents']:
                print(f"Found {test_course}: {result['documents'][0][:100]}...")
            else:
                print(f"WARNING: {test_course} not found in database!")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    init_database() 