import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_data, merge_datasets, save_processed_data
from src.utils.preprocessor import preprocess_movies, create_soup
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender


def evaluate_recommendations(recommender, test_movies: list, n: int = 10):
    print(f"\n{'='*60}")
    print(f"Evaluating recommendations for {len(test_movies)} movies")
    print(f"{'='*60}")
    
    successful = 0
    total_diversity = []
    
    for movie in test_movies:
        try:
            recommendations = recommender.get_recommendations(movie, n=n)
            successful += 1
            
            print(f"\n🎬 Recommendations for '{movie}':")
            for i, (title, score) in enumerate(recommendations[:5], 1):
                print(f"   {i}. {title} (score: {score:.3f})")
            
            # Calculate genre diversity (simple metric)
            rec_titles = [r[0] for r in recommendations]
            
        except ValueError as e:
            print(f"   ⚠️  Could not find '{movie}'")
    
    success_rate = successful / len(test_movies)
    
    print(f"\n{'='*60}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"{'='*60}\n")
    
    return {
        'success_rate': success_rate,
        'successful_recommendations': successful
    }


def compare_models(content_model, collab_model, test_movies: list):
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\n Content-Based Recommender:")
    content_metrics = evaluate_recommendations(content_model, test_movies)
    
    print("\n Collaborative Recommender:")
    collab_metrics = evaluate_recommendations(collab_model, test_movies)
    
    # Compare overlap
    print("\n Analyzing recommendation overlap...")
    movie = test_movies[0]
    
    try:
        content_recs = set([r[0] for r in content_model.get_recommendations(movie, n=10)])
        collab_recs = set([r[0] for r in collab_model.get_recommendations(movie, n=10)])
        
        overlap = content_recs.intersection(collab_recs)
        overlap_pct = len(overlap) / 10 * 100
        
        print(f" For '{movie}':")
        print(f" - Recommendation overlap: {len(overlap)}/10 ({overlap_pct:.0f}%)")
        print(f" - Unique to content-based: {len(content_recs - collab_recs)}")
        print(f" - Unique to collaborative: {len(collab_recs - content_recs)}")
    except:
        pass
    
    print("\n" + "="*80)


def main():
    """Main training pipeline."""
    
    print("="*80)
    print("MOVIE RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load data
    print("STEP 1: Loading data...")
    try:
        movies, credits = load_data('data/raw/')
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print("\n Please download the dataset from:")
        print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("And place the CSV files in 'data/raw/' directory\n")
        return
    
    # Step 2: Merge datasets
    print("\nSTEP 2: Merging datasets...")
    merged_df = merge_datasets(movies, credits)
    
    # Step 3: Preprocess data
    print("\nSTEP 3: Preprocessing data...")
    processed_df = preprocess_movies(merged_df)
    
    # Step 4: Create feature soup
    print("\nSTEP 4: Creating feature soup...")
    processed_df = create_soup(processed_df)
    
    # Save processed data
    save_processed_data(processed_df)
    
    print(f"\nProcessed {len(processed_df)} movies")
    print(f"Columns: {processed_df.columns.tolist()}")
    
    # Step 5: Train Content-Based Model
    print("\n" + "="*80)
    print("STEP 5: Training Content-Based Recommender")
    print("="*80)
    
    content_model = ContentBasedRecommender()
    content_model.fit(processed_df, soup_column='soup')
    content_model.save_model('models/content_based_model.pkl')
    
    # Step 6: Train Collaborative Model
    print("\n" + "="*80)
    print("STEP 6: Training Collaborative Recommender")
    print("="*80)
    
    collab_model = CollaborativeRecommender(n_neighbors=10, metric='cosine')
    collab_model.fit(processed_df)
    collab_model.save_model('models/collaborative_model.pkl')
    
    # Step 7: Evaluate models
    print("\n" + "="*80)
    print("STEP 7: Evaluating Models")
    print("="*80)
    
    # Test movies (popular and diverse genres)
    test_movies = [
        'The Dark Knight',
        'Inception',
        'The Shawshank Redemption',
        'Pulp Fiction',
        'Forrest Gump'
    ]
    
    compare_models(content_model, collab_model, test_movies)
    
    # Step 8: Generate sample recommendations
    print("\n" + "="*80)
    print("STEP 8: Sample Recommendations")
    print("="*80)
    
    sample_movie = 'The Dark Knight'
    print(f"\n Content-Based recommendations for '{sample_movie}':")
    
    try:
        recs = content_model.get_recommendations(sample_movie, n=10)
        for i, (title, score) in enumerate(recs, 1):
            print(f"   {i:2d}. {title:<50} (similarity: {score:.3f})")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved:")
    print(f"   - models/content_based_model.pkl")
    print(f"   - models/collaborative_model.pkl")
    print(f"\nProcessed data saved:")
    print(f"   - data/processed/movies_processed.csv")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()