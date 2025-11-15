import sys
import os
sys.path.insert(0, os.getcwd())

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict

print("="*90)
print("🎯 MODEL ACCURACY EVALUATION & COMPARISON")
print("="*90)

# Load models
print("\n📦 Loading models...")
content_model = ContentBasedRecommender.load_model('models/content_based_model.pkl')
collab_model = CollaborativeRecommender.load_model('models/collaborative_model.pkl')
all_movies = content_model.movies
print("✅ Models loaded successfully\n")

# Test movies
test_movies = [
    'The Dark Knight', 'Iron Man', 'The Avengers',
    'Inception', 'The Matrix', 'Interstellar',
    'The Shawshank Redemption', 'Forrest Gump', 'The Godfather',
    'Pulp Fiction', 'The Silence of the Lambs', 'Se7en',
    'Titanic', 'The Notebook',
    'Toy Story', 'Finding Nemo',
    'The Hangover', 'Superbad'
]

print(f"📊 Test Dataset: {len(test_movies)} movies\n")


def calculate_genre_match_accuracy(query_genres, rec_genres_list):
    """
    Calculate what % of recommendations share at least one genre with query.
    This is our PRIMARY ACCURACY METRIC.
    """
    matches = 0
    for rec_genres in rec_genres_list:
        if set(query_genres).intersection(set(rec_genres)):
            matches += 1
    return (matches / len(rec_genres_list) * 100) if rec_genres_list else 0


def calculate_rating_accuracy(query_rating, rec_ratings, tolerance=1.5):
    """
    Calculate what % of recommendations are within rating tolerance.
    """
    matches = 0
    for rec_rating in rec_ratings:
        if abs(query_rating - rec_rating) <= tolerance:
            matches += 1
    return (matches / len(rec_ratings) * 100) if rec_ratings else 0


def calculate_relevance_accuracy(similarity_scores, threshold=0.2):
    """
    Calculate what % of recommendations exceed similarity threshold.
    Content-Based: threshold=0.2
    Collaborative: threshold=0.95
    """
    above_threshold = sum(1 for score in similarity_scores if score >= threshold)
    return (above_threshold / len(similarity_scores) * 100) if similarity_scores else 0


def calculate_top_k_precision(recommendations, query_movie_data, k=5):
    """
    Precision@K: What % of top-K recommendations are "relevant"
    Relevant = shares at least 2 genres OR rating within 1 point
    """
    relevant = 0
    query_genres = set(query_movie_data['genres'])
    query_rating = query_movie_data['vote_average']
    
    for i, (title, score) in enumerate(recommendations[:k]):
        try:
            rec_data = all_movies[all_movies['title'] == title].iloc[0]
            rec_genres = set(rec_data['genres'])
            rec_rating = rec_data['vote_average']
            
            # Check if relevant
            genre_overlap = len(query_genres.intersection(rec_genres))
            rating_close = abs(query_rating - rec_rating) <= 1.0
            
            if genre_overlap >= 2 or rating_close:
                relevant += 1
        except:
            pass
    
    return (relevant / k * 100) if k > 0 else 0


class AccuracyEvaluator:
    """Evaluate model with accuracy metrics."""
    
    def __init__(self, model, model_name):
        self.model = model
        self.name = model_name
        self.similarity_threshold = 0.2 if 'Content' in model_name else 0.95
        
    def evaluate(self, test_movies, n_recommendations=10):
        """Run accuracy evaluation."""
        print(f"\n{'='*90}")
        print(f"📊 Evaluating: {self.name}")
        print('='*90)
        
        metrics = {
            'genre_accuracies': [],
            'rating_accuracies': [],
            'relevance_accuracies': [],
            'precision_at_5': [],
            'precision_at_10': [],
            'success_count': 0,
            'total_recommendations': 0,
            'response_times': []
        }
        
        for i, movie in enumerate(test_movies, 1):
            print(f"\r   Processing {i}/{len(test_movies)}: {movie[:40]:<40}", end='')
            
            try:
                # Get query movie data
                query_data = all_movies[all_movies['title'] == movie].iloc[0]
                query_genres = query_data['genres']
                query_rating = query_data['vote_average']
                
                # Get recommendations
                start = time.time()
                recommendations = self.model.get_recommendations(movie, n=n_recommendations)
                elapsed = (time.time() - start) * 1000
                
                metrics['response_times'].append(elapsed)
                metrics['success_count'] += 1
                
                # Extract recommendation data
                rec_genres_list = []
                rec_ratings = []
                similarity_scores = []
                
                for rec_title, score in recommendations:
                    similarity_scores.append(score)
                    try:
                        rec_data = all_movies[all_movies['title'] == rec_title].iloc[0]
                        rec_genres_list.append(rec_data['genres'])
                        rec_ratings.append(rec_data['vote_average'])
                    except:
                        pass
                
                # Calculate accuracy metrics
                genre_acc = calculate_genre_match_accuracy(query_genres, rec_genres_list)
                rating_acc = calculate_rating_accuracy(query_rating, rec_ratings, tolerance=1.5)
                relevance_acc = calculate_relevance_accuracy(similarity_scores, self.similarity_threshold)
                p_at_5 = calculate_top_k_precision(recommendations, query_data, k=5)
                p_at_10 = calculate_top_k_precision(recommendations, query_data, k=10)
                
                metrics['genre_accuracies'].append(genre_acc)
                metrics['rating_accuracies'].append(rating_acc)
                metrics['relevance_accuracies'].append(relevance_acc)
                metrics['precision_at_5'].append(p_at_5)
                metrics['precision_at_10'].append(p_at_10)
                metrics['total_recommendations'] += len(recommendations)
                
            except Exception as e:
                pass
        
        print()  # New line
        
        # Calculate final metrics
        self.metrics = {
            'success_rate': (metrics['success_count'] / len(test_movies)) * 100,
            
            # ACCURACY METRICS (our main focus)
            'genre_match_accuracy': np.mean(metrics['genre_accuracies']) if metrics['genre_accuracies'] else 0,
            'rating_accuracy': np.mean(metrics['rating_accuracies']) if metrics['rating_accuracies'] else 0,
            'relevance_accuracy': np.mean(metrics['relevance_accuracies']) if metrics['relevance_accuracies'] else 0,
            'precision_at_5': np.mean(metrics['precision_at_5']) if metrics['precision_at_5'] else 0,
            'precision_at_10': np.mean(metrics['precision_at_10']) if metrics['precision_at_10'] else 0,
            
            # Overall Accuracy (weighted average)
            'overall_accuracy': 0,  # Will calculate below
            
            # Performance
            'avg_response_time': np.mean(metrics['response_times']) if metrics['response_times'] else 0,
            'total_recommendations': metrics['total_recommendations']
        }
        
        # Calculate Overall Accuracy (weighted average of all accuracy metrics)
        self.metrics['overall_accuracy'] = (
            self.metrics['genre_match_accuracy'] * 0.40 +  # 40% weight - most important
            self.metrics['precision_at_10'] * 0.25 +        # 25% weight
            self.metrics['rating_accuracy'] * 0.20 +        # 20% weight
            self.metrics['relevance_accuracy'] * 0.15       # 15% weight
        )
        
        return self.metrics
    
    def print_summary(self):
        """Print summary with clear accuracy percentages."""
        m = self.metrics
        
        print(f"\n{'─'*90}")
        print(f"🎯 {self.name} - ACCURACY RESULTS")
        print('─'*90)
        
        print(f"\n📊 OVERALL ACCURACY: {m['overall_accuracy']:.2f}%")
        print("   (Weighted average of all accuracy metrics)")
        
        print(f"\n1️⃣  GENRE MATCH ACCURACY: {m['genre_match_accuracy']:.2f}%")
        print("   → What % of recommendations share genres with the query movie")
        print("   → HIGHER IS BETTER")
        
        print(f"\n2️⃣  RATING ACCURACY: {m['rating_accuracy']:.2f}%")
        print("   → What % of recommendations have similar ratings (±1.5 points)")
        print("   → HIGHER IS BETTER")
        
        print(f"\n3️⃣  RELEVANCE ACCURACY: {m['relevance_accuracy']:.2f}%")
        print("   → What % of recommendations exceed similarity threshold")
        print("   → HIGHER IS BETTER")
        
        print(f"\n4️⃣  PRECISION @ TOP-5: {m['precision_at_5']:.2f}%")
        print("   → What % of top 5 recommendations are highly relevant")
        print("   → HIGHER IS BETTER")
        
        print(f"\n5️⃣  PRECISION @ TOP-10: {m['precision_at_10']:.2f}%")
        print("   → What % of all 10 recommendations are relevant")
        print("   → HIGHER IS BETTER")
        
        print(f"\n6️⃣  RELIABILITY: {m['success_rate']:.2f}%")
        print("   → Success rate for generating recommendations")
        
        print(f"\n7️⃣  SPEED: {m['avg_response_time']:.2f}ms")
        print("   → Average response time")


def hybrid_recommend(movie, n=10):
    """Hybrid recommendation."""
    content_recs = content_model.get_recommendations(movie, n=n)
    collab_recs = collab_model.get_recommendations(movie, n=n)
    
    combined = {}
    for title, score in content_recs:
        combined[title] = score * 0.6
    for title, score in collab_recs:
        combined[title] = combined.get(title, 0) + score * 0.4
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]


class HybridModel:
    def get_recommendations(self, movie, n=10):
        return hybrid_recommend(movie, n)


# Evaluate all models
evaluators = [
    AccuracyEvaluator(content_model, "Content-Based"),
    AccuracyEvaluator(collab_model, "Collaborative Filtering"),
    AccuracyEvaluator(HybridModel(), "Hybrid (60% Content + 40% Collab)")
]

results = {}
for evaluator in evaluators:
    metrics = evaluator.evaluate(test_movies, n_recommendations=10)
    evaluator.print_summary()
    results[evaluator.name] = metrics


# ============================================================================
# ACCURACY COMPARISON TABLE
# ============================================================================

print(f"\n\n{'='*90}")
print("🏆 ACCURACY COMPARISON TABLE")
print('='*90)

print(f"\n{'Model':<40} {'Overall':<12} {'Genre':<12} {'Rating':<12} {'P@10':<12} {'Speed':<12}")
print('─'*90)

for name, m in results.items():
    print(f"{name:<40} "
          f"{m['overall_accuracy']:>6.2f}%    "
          f"{m['genre_match_accuracy']:>6.2f}%    "
          f"{m['rating_accuracy']:>6.2f}%    "
          f"{m['precision_at_10']:>6.2f}%    "
          f"{m['avg_response_time']:>6.2f}ms")


# ============================================================================
# WINNER ANALYSIS
# ============================================================================

print(f"\n\n{'='*90}")
print("🥇 ACCURACY WINNERS")
print('='*90)

accuracy_metrics = [
    ('Overall Accuracy', 'overall_accuracy'),
    ('Genre Match Accuracy', 'genre_match_accuracy'),
    ('Rating Accuracy', 'rating_accuracy'),
    ('Relevance Accuracy', 'relevance_accuracy'),
    ('Precision@5', 'precision_at_5'),
    ('Precision@10', 'precision_at_10')
]

print()
for metric_name, metric_key in accuracy_metrics:
    values = [(name, results[name][metric_key]) for name in results.keys()]
    winner = max(values, key=lambda x: x[1])
    print(f"🏆 {metric_name:30s}: {winner[0]:40s} {winner[1]:6.2f}%")


# ============================================================================
# DETAILED ACCURACY BREAKDOWN
# ============================================================================

print(f"\n\n{'='*90}")
print("📋 DETAILED ACCURACY BREAKDOWN")
print('='*90)

for name, m in results.items():
    print(f"\n{name}:")
    print("─" * 90)
    print(f"  Overall Accuracy:        {m['overall_accuracy']:6.2f}% ⭐")
    print(f"  ├─ Genre Match:          {m['genre_match_accuracy']:6.2f}% (40% weight)")
    print(f"  ├─ Precision@10:         {m['precision_at_10']:6.2f}% (25% weight)")
    print(f"  ├─ Rating Accuracy:      {m['rating_accuracy']:6.2f}% (20% weight)")
    print(f"  └─ Relevance:            {m['relevance_accuracy']:6.2f}% (15% weight)")


# ============================================================================
# ACCURACY GRADING
# ============================================================================

print(f"\n\n{'='*90}")
print("📝 ACCURACY GRADING")
print('='*90)

def get_grade(accuracy):
    """Convert accuracy to letter grade."""
    if accuracy >= 90: return "A+ (Excellent)"
    elif accuracy >= 80: return "A  (Very Good)"
    elif accuracy >= 70: return "B+ (Good)"
    elif accuracy >= 60: return "B  (Above Average)"
    elif accuracy >= 50: return "C  (Average)"
    else: return "D  (Below Average)"

print()
for name, m in results.items():
    grade = get_grade(m['overall_accuracy'])
    print(f"{name:40s}: {m['overall_accuracy']:6.2f}% → {grade}")


# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================

print(f"\n\n{'='*90}")
print("💡 FINAL RECOMMENDATION")
print('='*90)

best_model = max(results.items(), key=lambda x: x[1]['overall_accuracy'])

print(f"""
🏆 BEST MODEL: {best_model[0]}
   Overall Accuracy: {best_model[1]['overall_accuracy']:.2f}%
   Grade: {get_grade(best_model[1]['overall_accuracy'])}

📊 Why this model wins:
   • Genre Match: {best_model[1]['genre_match_accuracy']:.2f}%
   • Precision@10: {best_model[1]['precision_at_10']:.2f}%
   • Rating Accuracy: {best_model[1]['rating_accuracy']:.2f}%
   • Speed: {best_model[1]['avg_response_time']:.2f}ms

✅ RECOMMENDATION: Use {best_model[0]} for production deployment
""")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"{'='*90}")
print("💾 SAVING RESULTS")
print('='*90)

os.makedirs('docs', exist_ok=True)

# Save to Markdown
with open('docs/ACCURACY_RESULTS.md', 'w') as f:
    f.write("# Model Accuracy Comparison Results\n\n")
    f.write(f"**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Test Dataset:** {len(test_movies)} diverse movies\n\n")
    
    f.write("## Accuracy Summary\n\n")
    f.write("| Model | Overall Accuracy | Genre Match | Rating Match | Precision@10 |\n")
    f.write("|-------|-----------------|-------------|--------------|---------------|\n")
    for name, m in results.items():
        f.write(f"| {name} | {m['overall_accuracy']:.2f}% | {m['genre_match_accuracy']:.2f}% | "
                f"{m['rating_accuracy']:.2f}% | {m['precision_at_10']:.2f}% |\n")
    
    f.write(f"\n## Winner\n\n")
    f.write(f"**{best_model[0]}** achieves the highest overall accuracy of **{best_model[1]['overall_accuracy']:.2f}%**\n\n")
    
    f.write("## Accuracy Metrics Explained\n\n")
    f.write("- **Genre Match Accuracy**: % of recommendations sharing genres with query movie\n")
    f.write("- **Rating Accuracy**: % of recommendations within ±1.5 rating points\n")
    f.write("- **Precision@10**: % of top-10 recommendations that are relevant\n")
    f.write("- **Overall Accuracy**: Weighted average of all metrics\n")

print("✅ Results saved to: docs/ACCURACY_RESULTS.md")

# Save to CSV
df_results = pd.DataFrame([
    {
        'Model': name,
        'Overall_Accuracy_%': m['overall_accuracy'],
        'Genre_Match_%': m['genre_match_accuracy'],
        'Rating_Accuracy_%': m['rating_accuracy'],
        'Relevance_Accuracy_%': m['relevance_accuracy'],
        'Precision_at_5_%': m['precision_at_5'],
        'Precision_at_10_%': m['precision_at_10'],
        'Speed_ms': m['avg_response_time']
    }
    for name, m in results.items()
])

df_results.to_csv('docs/accuracy_metrics.csv', index=False)
print("✅ Detailed metrics saved to: docs/accuracy_metrics.csv")

print(f"\n{'='*90}")
print("✅ ACCURACY EVALUATION COMPLETE!")
print('='*90)
print(f"\n🎯 Summary:")
print(f"   • {len(test_movies)} movies tested")
print(f"   • 3 models evaluated")
print(f"   • 6 accuracy metrics calculated")
print(f"   • Winner: {best_model[0]} ({best_model[1]['overall_accuracy']:.2f}%)")
print('='*90 + "\n")