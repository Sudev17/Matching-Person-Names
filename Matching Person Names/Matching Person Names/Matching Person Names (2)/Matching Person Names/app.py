from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from names_data import NAMES
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and embeddings
model = None
name_embeddings = None

def initialize_model():
    global model, name_embeddings
    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Pre-computing embeddings for all names...")
    name_embeddings = model.encode(NAMES, convert_to_numpy=True, normalize_embeddings=True)
    print(f"Pre-computed embeddings for {len(NAMES)} names.")

def ensure_model():
    global model, name_embeddings
    if model is None or name_embeddings is None:
        initialize_model()

def preprocess_name(name):
    """Preprocess name for better matching"""
    return name.strip().lower()

def get_matches(query_name, top_k=10):
    ensure_model()
    processed_query = preprocess_name(query_name)
    query_embedding = model.encode([processed_query], convert_to_numpy=True, normalize_embeddings=True)
    similarities = np.dot(name_embeddings, query_embedding.T).flatten()
    for i, name in enumerate(NAMES):
        name_lower = name.lower()
        if processed_query in name_lower or name_lower in processed_query:
            similarities[i] *= 1.1
    sorted_indices = np.argsort(similarities)[::-1]
    best_idx = sorted_indices[0]
    best_match = NAMES[best_idx]
    best_score = float(similarities[best_idx])
    top_matches = []
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        top_matches.append({
            "name": NAMES[idx],
            "score": float(similarities[idx])
        })
    return {
        "best_match": {
            "name": best_match,
            "score": best_score
        },
        "top_matches": top_matches
    }

 

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/match', methods=['POST'])
def match_name():
    try:
        data = request.get_json()
        query_name = data.get('name', '').strip()
        
        if not query_name:
            return jsonify({"error": "Please provide a name to search"}), 400
        
        results = get_matches(query_name)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
