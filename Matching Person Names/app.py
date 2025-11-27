from flask import Flask, render_template, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
from names_data import NAMES
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and embeddings
model = None
name_embeddings = None

def initialize_model():
    """Initialize the embedding model and pre-compute embeddings"""
    global model, name_embeddings
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Pre-computing embeddings for all names...")
    name_embeddings = model.encode(NAMES, convert_to_numpy=True, normalize_embeddings=True)
    print(f"Pre-computed embeddings for {len(NAMES)} names.")

def preprocess_name(name):
    """Preprocess name for better matching"""
    return name.strip().lower()

def get_matches(query_name, top_k=10):
    """Get matches for a query name"""
    if model is None or name_embeddings is None:
        return {"error": "Model not initialized"}
    
    # Preprocess the query name
    processed_query = preprocess_name(query_name)
    
    # Embed the query name
    query_embedding = model.encode([processed_query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Compute cosine similarity via dot-product (since embeddings are normalized)
    similarities = np.dot(name_embeddings, query_embedding.T).flatten()
    
    # Apply a small boost to exact substring matches
    for i, name in enumerate(NAMES):
        name_lower = name.lower()
        if processed_query in name_lower or name_lower in processed_query:
            similarities[i] *= 1.1  # Boost by 10%
    
    # Sort by similarity scores (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Get the best match
    best_idx = sorted_indices[0]
    best_match = NAMES[best_idx]
    best_score = float(similarities[best_idx])
    
    # Get top_k matches
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

# Initialize model when app starts
with app.app_context():
    initialize_model()

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