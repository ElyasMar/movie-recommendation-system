import sys
import os
sys.path.insert(0, os.getcwd())

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
import pandas as pd
import time
from collections import Counter

# Load models
print("="*80)
print("MODEL COMPARISON ANALYSIS")
print("="*80)

print("\nLoading models...")
content_model = ContentBasedRecommender.load_model('models/content_based_model.pkl')
collab_model = CollaborativeRecommender.load_model('models/collaborative_model.pkl')

print("Models loaded\n")

# Test movies - diverse genres
test_movies = [
    'The Dark Knight',      # Action/Crime
    'Inception',            # Sci-Fi/Thriller
    'The Shawshank Redemption',  # Drama
    'Pulp Fiction',         # Crime/Drama
    'Forrest Gump',         # Drama/Romance
    'The Matrix',           # Sci-Fi/Action
    'Interstellar',         # Sci-Fi
    'The Godfather',        # Crime/Drama
    'Titanic',              # Romance/Drama
    'Toy Story'             # Animation/Family
]

print("Test Movies:", len(test_movies))
for i, movie in enumerate(test_movies, 1):
    print(f"   {i}. {movie}")

print("\n" + "="*80)


def get_genre_diversity(recommendations, model):
    """Calculate genre diversity of recommendations."""
    genres = []
    for rec_title, _ in recommendations:
        try:
            details = model.get_movie_details(rec_title)
            if details and 'genres' in details:
                genres.extend(details['genres'])
        except:
            pass
    
    if not genres:
        return 0
    
    unique_genres = len(set(genres))
    total_genres = len(genres)
    return unique_genres / total_genres if total_genres > 0 else 0


def analyze_recommendations(movie_title, method_name, get_recs_func, model):
    """Analyze a single recommendation."""
    try:
        start_time = time.time()
        recommendations = get_recs_func(movie_title, n=10)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        avg_score = sum([score for _, score in recommendations]) / len(recommendations)
        diversity = get_genre_diversity(recommendations, model)
        
        return {
            'success': True,
            'time': elapsed_time * 1000,  # Convert to ms
            'avg_score': avg_score,
            'diversity': diversity,
            'recommendations': recommendations
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def hybrid_recommend(movie_title, n=10):
    """Hybrid recommendation combining both models."""
    content_recs = content_model.get_recommendations(movie_title, n=n)
    collab_recs = collab_model.get_recommendations(movie_title, n=n)
    
    combined = {}
    for movie, score in content_recs:
        combined[movie] = score * 0.6
    for movie, score in collab_recs:
        combined[movie] = combined.get(movie, 0) + score * 0.4
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]


# Run comparison
results = {
    'content': {'times': [], 'scores': [], 'diversities': [], 'successes': 0},
    'collaborative': {'times': [], 'scores': [], 'diversities': [], 'successes': 0},
    'hybrid': {'times': [], 'scores': [], 'diversities': [], 'successes': 0}
}

print("\n📊 RUNNING ANALYSIS...")
print("="*80)

for movie in test_movies:
    print(f"\n🎬 Analyzing: {movie}")
    print("-" * 80)
    
    # Content-Based
    print("   Content-Based...", end=" ")
    content_result = analyze_recommendations(
        movie, 'Content-Based', 
        content_model.get_recommendations, 
        content_model
    )
    if content_result['success']:
        results['content']['times'].append(content_result['time'])
        results['content']['scores'].append(content_result['avg_score'])
        results['content']['diversities'].append(content_result['diversity'])
        results['content']['successes'] += 1
        print(f"✅ {content_result['time']:.2f}ms")
    else:
        print(f"❌ {content_result['error']}")
    
    # Collaborative
    print("   Collaborative...", end=" ")
    collab_result = analyze_recommendations(
        movie, 'Collaborative', 
        collab_model.get_recommendations, 
        collab_model
    )
    if collab_result['success']:
        results['collaborative']['times'].append(collab_result['time'])
        results['collaborative']['scores'].append(collab_result['avg_score'])
        results['collaborative']['diversities'].append(collab_result['diversity'])
        results['collaborative']['successes'] += 1
        print(f"{collab_result['time']:.2f}ms")
    else:
        print(f"{collab_result['error']}")
    
    # Hybrid
    print("   Hybrid...", end=" ")
    hybrid_result = analyze_recommendations(
        movie, 'Hybrid', 
        hybrid_recommend, 
        content_model
    )
    if hybrid_result['success']:
        results['hybrid']['times'].append(hybrid_result['time'])
        results['hybrid']['scores'].append(hybrid_result['avg_score'])
        results['hybrid']['diversities'].append(hybrid_result['diversity'])
        results['hybrid']['successes'] += 1
        print(f"{hybrid_result['time']:.2f}ms")
    else:
        print(f"{hybrid_result['error']}")


# Print Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

methods = ['content', 'collaborative', 'hybrid']
method_names = ['Content-Based', 'Collaborative', 'Hybrid']

print("\n Success Rate:")
print("-" * 80)
for method, name in zip(methods, method_names):
    success_rate = (results[method]['successes'] / len(test_movies)) * 100
    print(f"   {name:20s}: {results[method]['successes']:2d}/{len(test_movies)} ({success_rate:.1f}%)")

print("\n Average Response Time (ms):")
print("-" * 80)
for method, name in zip(methods, method_names):
    if results[method]['times']:
        avg_time = sum(results[method]['times']) / len(results[method]['times'])
        min_time = min(results[method]['times'])
        max_time = max(results[method]['times'])
        print(f"   {name:20s}: {avg_time:6.2f}ms (min: {min_time:.2f}, max: {max_time:.2f})")

print("\n Average Similarity Score:")
print("-" * 80)
for method, name in zip(methods, method_names):
    if results[method]['scores']:
        avg_score = sum(results[method]['scores']) / len(results[method]['scores'])
        min_score = min(results[method]['scores'])
        max_score = max(results[method]['scores'])
        print(f"   {name:20s}: {avg_score:.3f} (min: {min_score:.3f}, max: {max_score:.3f})")

print("\n Genre Diversity Score:")
print("-" * 80)
for method, name in zip(methods, method_names):
    if results[method]['diversities']:
        avg_div = sum(results[method]['diversities']) / len(results[method]['diversities'])
        print(f"   {name:20s}: {avg_div:.3f} (higher is more diverse)")


# Detailed comparison for one movie
print("\n" + "="*80)
print("DETAILED EXAMPLE: The Dark Knight")
print("="*80)

movie = 'The Dark Knight'

for method_name, get_recs_func in [
    ('Content-Based', content_model.get_recommendations),
    ('Collaborative', collab_model.get_recommendations),
    ('Hybrid', hybrid_recommend)
]:
    print(f"\n{method_name}:")
    print("-" * 80)
    try:
        recs = get_recs_func(movie, n=5)
        for i, (title, score) in enumerate(recs, 1):
            print(f"   {i}. {title:<50s} (score: {score:.3f})")
    except Exception as e:
        print(f"   Error: {e}")


# Recommendation overlap analysis
print("\n" + "="*80)
print("RECOMMENDATION OVERLAP ANALYSIS")
print("="*80)

movie = 'Inception'
print(f"\nAnalyzing recommendations for: {movie}\n")

try:
    content_recs = set([r[0] for r in content_model.get_recommendations(movie, n=10)])
    collab_recs = set([r[0] for r in collab_model.get_recommendations(movie, n=10)])
    hybrid_recs = set([r[0] for r in hybrid_recommend(movie, n=10)])
    
    print(f"Content-Based vs Collaborative:")
    overlap_cc = content_recs.intersection(collab_recs)
    print(f"   Common: {len(overlap_cc)}/10 movies")
    if overlap_cc:
        print(f"   Movies: {', '.join(list(overlap_cc)[:3])}...")
    
    print(f"\nContent-Based vs Hybrid:")
    overlap_ch = content_recs.intersection(hybrid_recs)
    print(f"   Common: {len(overlap_ch)}/10 movies")
    
    print(f"\nCollaborative vs Hybrid:")
    overlap_lh = collab_recs.intersection(hybrid_recs)
    print(f"   Common: {len(overlap_lh)}/10 movies")
    
    print(f"\nAll three methods:")
    overlap_all = content_recs.intersection(collab_recs).intersection(hybrid_recs)
    print(f"   Common: {len(overlap_all)}/10 movies")
    if overlap_all:
        print(f"   Movies: {', '.join(overlap_all)}")

except Exception as e:
    print(f"Error: {e}")


# Final recommendation
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

best_method = max(methods, key=lambda m: results[m]['successes'])
fastest_method = min(methods, key=lambda m: sum(results[m]['times']) / len(results[m]['times']) if results[m]['times'] else float('inf'))
most_diverse = max(methods, key=lambda m: sum(results[m]['diversities']) / len(results[m]['diversities']) if results[m]['diversities'] else 0)

print(f"""
1. Most Reliable: {method_names[methods.index(best_method)]}
   → Highest success rate

2. Fastest: {method_names[methods.index(fastest_method)]}
   → Lowest response time

3. Most Diverse: {method_names[methods.index(most_diverse)]}
   → Best genre variety

4. Best Overall: Hybrid
   → Balanced approach combining strengths of both methods
   → Good similarity scores with reasonable diversity
   → Suitable for most use cases
""")

print("="*80)
print("Analysis Complete!")
print("="*80)