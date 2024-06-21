import random
import json
import nltk
from flask import Flask, request, jsonify

# Load the processed data
with open('processed_data.json', 'r') as f:
    data = json.load(f)

input_texts = data['input_texts']
target_texts = data['target_texts']

# Initialize Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = get_response(user_message)
    return jsonify({'response': response})

def get_response(user_message):
    tokenized_user_message = nltk.word_tokenize(user_message.lower())
    best_match_index = random.randint(0, len(input_texts) - 1)  # Dummy response selection
    return target_texts[best_match_index]

if __name__ == '__main__':
    app.run(debug=True)
