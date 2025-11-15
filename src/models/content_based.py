import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import joblib
from typing import List, Tuple


class ContentBasedRecommender:
    
    def __init__(self):
        """Initialize the content-based recommender."""
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.movies = None
        
    
    def fit(self, df: pd.DataFrame, soup_column: str = 'soup'):
        print("Fitting Content-Based Recommender...")
        
        self.movies = df.copy()
        
        # Initialize TF-IDF Vectorizer
        # max_features: limit vocabulary size
        # stop_words: remove common words
        # ngram_range: use unigrams and bigrams
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the soup
        print("Computing TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf.fit_transform(df[soup_column])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Compute cosine similarity
        print("Computing cosine similarity...")
        # Use linear_kernel instead of cosine_similarity (faster for TF-IDF)
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create reverse mapping of indices
        self.indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        print("Content-Based Recommender fitted successfully")
        
        return self
    
    
    def get_recommendations(self, title: str, n: int = 10) -> List[Tuple[str, float]]:
        # Check if title exists
        if title not in self.indices:
            available_titles = self.indices.index.tolist()[:10]
            raise ValueError(
                f"Movie '{title}' not found in database. "
                f"Available titles include: {available_titles}"
            )
        
        # Get the index of the movie
        idx = self.indices[title]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get movie indices and scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return titles and scores
        recommended_movies = self.movies.iloc[movie_indices]['title'].tolist()
        
        return list(zip(recommended_movies, similarity_scores))
    
    
    def get_movie_details(self, title: str) -> dict:
        if title not in self.indices:
            return None
        
        idx = self.indices[title]
        movie = self.movies.iloc[idx]
        
        return {
            'title': movie['title'],
            'overview': movie['overview'],
            'genres': movie['genres'],
            'keywords': movie['keywords'],
            'cast': movie['cast'],
            'director': movie['director'],
            'vote_average': movie['vote_average'],
            'vote_count': int(movie['vote_count']),
            'popularity': movie['popularity']
        }
    
    
    def save_model(self, path: str = 'models/content_based_model.pkl'):
        model_data = {
            'tfidf': self.tfidf,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'movies': self.movies
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    
    @classmethod
    def load_model(cls, path: str = 'models/content_based_model.pkl'):
        model_data = joblib.load(path)
        
        recommender = cls()
        recommender.tfidf = model_data['tfidf']
        recommender.tfidf_matrix = model_data['tfidf_matrix']
        recommender.cosine_sim = model_data['cosine_sim']
        recommender.indices = model_data['indices']
        recommender.movies = model_data['movies']
        
        print(f"Model loaded from {path}")
        
        return recommender