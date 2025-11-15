from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
content_model = None
collab_model = None


def load_models():
    """Load trained models from disk."""
    global content_model, collab_model
    
    if content_model is None:
        print("Loading content-based model...")
        model_path = os.path.join(project_root, 'models', 'content_based_model.pkl')
        content_model = ContentBasedRecommender.load_model(model_path)
        print("✅ Content-based model loaded")
    
    if collab_model is None:
        print("Loading collaborative model...")
        model_path = os.path.join(project_root, 'models', 'collaborative_model.pkl')
        collab_model = CollaborativeRecommender.load_model(model_path)
        print("✅ Collaborative model loaded")


# Define routes BEFORE the main block
@app.route('/', methods=['GET'])
def home():
    """Root endpoint."""
    return jsonify({
        'message': 'Movie Recommendation API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            'GET /',
            'GET /api/health',
            'GET /api/movies',
            'GET /api/movies/<title>',
            'POST /api/recommend',
            'GET /api/search'
        ]
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running',
        'version': '1.0.0'
    })


@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies."""
    try:
        load_models()
        
        query = request.args.get('q', '').lower()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Missing parameter: q'}), 400
        
        movies_df = content_model.movies
        matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
        matches = matches.head(limit)
        
        results = []
        for _, row in matches.iterrows():
            results.append({
                'title': row['title'],
                'overview': row['overview'][:150] + '...' if len(str(row['overview'])) > 150 else str(row['overview']),
                'genres': row['genres'],
                'vote_average': float(row['vote_average'])
            })
        
        return jsonify({
            'query': query,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/movies', methods=['GET'])
def list_movies():
    """List movies."""
    try:
        load_models()
        
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        movies_df = content_model.movies
        total = len(movies_df)
        movies_df = movies_df.iloc[offset:offset+limit]
        
        movies = []
        for _, row in movies_df.iterrows():
            movies.append({
                'title': row['title'],
                'vote_average': float(row['vote_average']),
                'genres': row['genres']
            })
        
        return jsonify({
            'total': total,
            'limit': limit,
            'offset': offset,
            'movies': movies
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/movies/<path:title>', methods=['GET'])
def get_movie_details(title):
    """Get movie details."""
    try:
        load_models()
        details = content_model.get_movie_details(title)
        
        if details is None:
            return jsonify({'error': f"Movie '{title}' not found"}), 404
        
        return jsonify(details)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get recommendations."""
    try:
        load_models()
        
        data = request.get_json()
        if not data or 'title' not in data:
            return jsonify({'error': 'Missing field: title'}), 400
        
        title = data['title']
        method = data.get('method', 'content')
        n = int(data.get('n', 10))
        
        if method not in ['content', 'collaborative', 'hybrid']:
            return jsonify({'error': 'Invalid method'}), 400
        
        # Get recommendations
        if method == 'content':
            recommendations = content_model.get_recommendations(title, n=n)
        elif method == 'collaborative':
            recommendations = collab_model.get_recommendations(title, n=n)
        else:  # hybrid
            content_recs = content_model.get_recommendations(title, n=n)
            collab_recs = collab_model.get_recommendations(title, n=n)
            
            combined = {}
            for movie, score in content_recs:
                combined[movie] = score * 0.6
            for movie, score in collab_recs:
                combined[movie] = combined.get(movie, 0) + score * 0.4
            
            recommendations = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Format results
        results = []
        for movie_title, score in recommendations:
            details = content_model.get_movie_details(movie_title)
            if details:
                results.append({
                    'title': movie_title,
                    'score': float(score),
                    'overview': details['overview'][:200] + '...' if len(details['overview']) > 200 else details['overview'],
                    'genres': details['genres'],
                    'vote_average': details['vote_average']
                })
        
        return jsonify({
            'query': title,
            'method': method,
            'count': len(results),
            'recommendations': results
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MOVIE RECOMMENDATION API")
    print("="*70)
    
    try:
        print("\nLoading models...")
        load_models()
        
        print("\nModels loaded successfully!")
        print("\nAvailable endpoints:")
        print("GET  /")
        print("GET  /api/health")
        print("GET  /api/search?q=<query>")
        print("GET  /api/movies")
        print("GET  /api/movies/<title>")
        print("POST /api/recommend")
        
        print("\n" + "="*70)
        print("Starting server on http://0.0.0.0:5000")
        print("="*70 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure models are trained: python src/train.py")
        sys.exit(1)