import os
import json
import pandas as pd
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

# Get the existing collection
try:
    collection = chroma_client.get_collection(
        name="courses",
        embedding_function=openai_ef
    )
    print("Successfully connected to existing collection")
except Exception as e:
    print(f"Error connecting to collection: {e}")
    print("Please run init_db.py first to initialize the database")
    exit(1)

def get_relevant_courses(query, n_results=3):
    """Retrieve relevant courses based on query"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def create_prompt(query, relevant_courses):
    """Create a prompt combining user query and relevant course information"""
    context = "\nRelevant courses:\n"
    for i, doc in enumerate(relevant_courses['documents'][0]):
        context += f"\n{doc}\n"
    
    prompt = f"""You are an AI academic advisor at Ohio State University's Computer Science department. 
A student has asked: "{query}"

Based on the following course information, provide a helpful response:
{context}

Please provide a clear, concise response that directly addresses the student's question using the course information provided. 
If discussing prerequisites, be specific about course numbers and requirements.
If the student's question isn't directly related to the courses shown, you can provide general academic advice while mentioning relevant courses.

Response:"""
    
    return prompt

def query_openai(prompt):
    """Query OpenAI with the constructed prompt"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful academic advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again later."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print(f"Received message: {user_input}")
    
    # Get relevant courses
    relevant_courses = get_relevant_courses(user_input)
    
    # Create prompt with context
    prompt = create_prompt(user_input, relevant_courses)
    
    # Get response from OpenAI
    response = query_openai(prompt)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)