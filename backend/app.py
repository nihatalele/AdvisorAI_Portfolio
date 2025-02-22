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
from werkzeug.utils import secure_filename
import PyPDF2

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Add these configurations after the CORS setup
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def extract_courses_from_pdf(file_path):
    """Extract text from PDF and identify course information"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Check if text was extracted
        if not text.strip():
            return "Error: No text extracted from the PDF."
        
        # Create a prompt for OpenAI to extract course information
        prompt = f"""Please analyze this transcript text and extract all CSE (Computer Science) courses. 
        Format the response as a list of course numbers only (e.g., CSE 1223, CSE 2221, etc.).
        Transcript text: {text}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a transcript analysis assistant. Extract only CSE course numbers from the transcript."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

@app.route('/upload-transcript', methods=['POST'])
def upload_transcript():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract courses from PDF
            courses = extract_courses_from_pdf(file_path)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            if courses:
                # Get course information for each extracted course
                course_info = []
                course_numbers = re.findall(r'CSE\s+\d{4}', courses)
                
                for course_num in course_numbers:
                    relevant_courses = get_relevant_courses(course_num, n_results=1)
                    if relevant_courses['documents'][0]:
                        course_info.append(relevant_courses['documents'][0][0])
                
                response_text = f"""Based on your transcript, I can see you've taken the following courses:\n\n{courses}\n\n
                Would you like specific information about any of these courses or recommendations for future courses based on your academic history?"""
                
                return jsonify({
                    "response": response_text,
                    "courses": course_info
                })
            
            return jsonify({"error": "Could not extract course information from the PDF"}), 400
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)