import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from names_data import NAMES
import warnings
warnings.filterwarnings("ignore")

def preprocess_name(name):
    """Preprocess name for better matching"""
    # Convert to lowercase and remove extra spaces
    name = name.strip().lower()
    # Could add more preprocessing like removing special characters if needed
    return name

def get_matches(query_name, names, name_embeddings, model, top_k=10):
    """
    Find the best matching names for a given query using cosine similarity.
    
    Args:
        query_name (str): The name to search for
        names (list): List of all names in the dataset
        name_embeddings (np.array): Precomputed embeddings for all names
        model (SentenceTransformer): The embedding model
        top_k (int): Number of top matches to return
        
    Returns:
        tuple: (best_match_name, best_similarity_score, top_matches_list)
    """
    # Preprocess the query name
    processed_query = preprocess_name(query_name)
    
    # Embed the query name
    query_embedding = model.encode([processed_query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Compute cosine similarity via dot-product (since embeddings are normalized)
    similarities = np.dot(name_embeddings, query_embedding.T).flatten()
    
    # Apply a small boost to exact substring matches
    for i, name in enumerate(names):
        name_lower = name.lower()
        if processed_query in name_lower or name_lower in processed_query:
            similarities[i] *= 1.1  # Boost by 10%
    
    # Sort by similarity scores (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Get the best match
    best_idx = sorted_indices[0]
    best_match = names[best_idx]
    best_score = similarities[best_idx]
    
    # Get top_k matches
    top_matches = []
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        top_matches.append((names[idx], similarities[idx]))
    
    return best_match, best_score, top_matches

def initialize_model():
    """Initialize and return the embedding model with optimizations"""
    print("Loading enhanced embedding model...")
    # Using a more powerful model for better embeddings
    model = SentenceTransformer('all-mpnet-base-v2')  # More powerful than L6-v2
    return model

def compute_embeddings(model, names):
    """Compute embeddings with batching for better performance"""
    print("Pre-computing enhanced embeddings for all names...")
    # Process in batches for better memory management
    embeddings = model.encode(
        names, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        batch_size=32  # Process in batches
    )
    return embeddings

def main():
    # Initialize embedding model
    model = initialize_model()
    
    # Pre-compute embeddings for all names
    name_embeddings = compute_embeddings(model, NAMES)
    print(f"Pre-computed embeddings for {len(NAMES)} names.")
    
    print("\n=== Enhanced Name Matching System ===")
    print("Features: Improved embeddings, preprocessing, and matching")
    print("Enter 'exit' to quit the program.\n")
    
    while True:
        # Get user input
        query_name = input("Enter a name to search: ").strip()
        
        # Exit condition
        if query_name.lower() == 'exit':
            print("Goodbye!")
            break
            
        if not query_name:
            print("Please enter a valid name.\n")
            continue
        
        # Get matches
        best_match, best_score, top_matches = get_matches(
            query_name, NAMES, name_embeddings, model, top_k=10
        )
        
        # Display results with enhanced formatting
        print(f"\n🎯 Best match:")
        print(f"   {best_match} (Similarity: {best_score:.4f})\n")
        
        print("🏆 Top matches:")
        for i, (name, score) in enumerate(top_matches, 1):
            bar_length = int(score * 20)  # Scale similarity to bar length
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {i:2d}. {name:<20} | {bar} | {score:.4f}")
        print()

if __name__ == "__main__":
    main()