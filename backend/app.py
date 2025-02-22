import os
import json
import pandas as pd
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from werkzeug.utils import secure_filename
import PyPDF2 

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

with open('/Users/Siddarth/AdvisorAI/backend/cse_courses.json') as f:
    courses = json.load(f)

df = pd.DataFrame(courses)

conversation_history = [{"role": "system", "content": "You are a helpful academic advisor."}]

# Clean and process the data
df['title'] = df['title'].str.strip()
df['description'] = df['description'].str.strip()
df['prerequisites'] = df['prerequisites'].str.strip()
df['credits'] = df['units'].str.extract('(\d+)').astype(int)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df['title'].tolist() + df['description'].tolist()) 

client = OpenAI(api_key='API_KEY')


def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def query_openai_for_classes(pdf_text):
    response = client.chat_completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an academic advisor analyzing completed classes."},
            {"role": "user", "content": f"Here is a student's transcript: {pdf_text}. List all the completed courses."}
        ]
    )
    return response.choices[0].message.content


@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        pdf_text = extract_text_from_pdf(file_path)

        conversation_history.append({"role": "user", "content": f"Here is my transcript: {pdf_text}"})

        openai_response = query_openai_for_classes(pdf_text)

        conversation_history.append({"role": "assistant", "content": openai_response})

        return jsonify({"response": openai_response}), 200
    
    return jsonify({"error": "File upload failed"}), 500


def query_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content

def get_course_info(course_number):
    course_info = df[df['number'] == course_number].iloc[0]
    return {
        "title": course_info['title'],
        "description": course_info['description'],
        "credits": course_info['credits'],
        "prerequisites": course_info['prerequisites'],
        "number": course_info['number'] 
    }

def extract_course_number(input_text):
    match = re.search(r'(CSE)\s*(\d{4})', input_text.upper())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

def find_best_matching_course(user_input):
    user_vector = vectorizer.transform([user_input])
    
    vectors = vectorizer.transform(df['title'].tolist() + df['description'].tolist())
    cosine_similarities = cosine_similarity(user_vector, vectors)
    
    best_match_index = np.argmax(cosine_similarities[0])
    
    if best_match_index < len(df):
        matched_course_number = df.iloc[best_match_index]['number']
        return get_course_info(matched_course_number)
    else:
        return None

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print(f"Received message: {user_input}")
    
    course_number = extract_course_number(user_input)
    
    if course_number:
            course_info = get_course_info(course_number)
            prompt = (
                f"{user_input} Here are the details about {course_info['title']} (CSE {course_number}): "
                f"{course_info['description']} This course is worth {course_info['credits']} credits. "
                f"The prerequisites are: {course_info['prerequisites']}."
                f"Answer ONLY the following prompt, in 1-2 sentences and don't include anything the following prompt didn't ask for. You are an AI academic advisor, so answer in a nice way like an academic advisor"
            )
    else:
        matched_course_info = find_best_matching_course(user_input)
        if matched_course_info:
            prompt = (
                f"{user_input} Based on your interest, I recommend {matched_course_info['title']} (CSE {matched_course_info['number']}): "
                f"{matched_course_info['description']} This course is worth {matched_course_info['credits']} credits. "
                f"The prerequisites are: {matched_course_info['prerequisites']}."
                f"Answer ONLY the following prompt, in 1-2 sentences and don't include anything the following prompt didn't ask for. You are an AI academic advisor, so answer in a nice way like an academic advisor"
            )
        else:
            prompt = "I'm sorry, but I couldn't find any relevant courses based on your query."
    



    conversation_history.append({"role": "user", "content": user_input})
    
    conversation_history.append({"role": "user", "content": prompt})

    response = query_openai(conversation_history)

    conversation_history.append({"role": "assistant", "content": response})
    
    response_json = jsonify({"response": response})
    return response_json



# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)