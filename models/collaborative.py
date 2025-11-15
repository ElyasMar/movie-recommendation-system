"""
Collaborative Filtering Recommender.

Uses user-item interactions and K-Nearest Neighbors to recommend movies
based on similar user preferences.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib
from typing import List, Tuple


class CollaborativeRecommender:
    
    def __init__(self, n_neighbors: int = 10, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None
        self.movie_features = None
        self.movies = None
        self.indices = None
    
    
    def fit(self, df: pd.DataFrame):
        print("Fitting Collaborative Recommender...")
        
        self.movies = df.copy()
        
        # Create feature matrix
        # Normalize features to same scale
        features = df[['popularity', 'vote_average', 'vote_count']].copy()
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Convert to sparse matrix for efficiency
        self.movie_features = csr_matrix(features_normalized)
        
        # Initialize and fit KNN model
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because query point is included
            metric=self.metric,
            algorithm='brute'
        )
        
        self.model.fit(self.movie_features)
        
        # Create reverse mapping
        self.indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        print("Collaborative Recommender fitted successfully")
        
        return self
    
    
    def get_recommendations(self, title: str, n: int = 10) -> List[Tuple[str, float]]:
        if title not in self.indices:
            available_titles = self.indices.index.tolist()[:10]
            raise ValueError(
                f"Movie '{title}' not found in database. "
                f"Available titles include: {available_titles}"
            )
        
        # Get movie index
        idx = self.indices[title]
        
        # Get movie features
        movie_feature = self.movie_features[idx]
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(
            movie_feature,
            n_neighbors=n + 1
        )
        
        # Get movie indices (excluding the query movie itself)
        movie_indices = indices.flatten()[1:]
        distance_scores = distances.flatten()[1:]
        
        # Convert distances to similarity scores (1 - distance for cosine)
        if self.metric == 'cosine':
            similarity_scores = 1 - distance_scores
        else:
            # For other metrics, use inverse distance
            similarity_scores = 1 / (1 + distance_scores)
        
        # Get movie titles
        recommended_movies = self.movies.iloc[movie_indices]['title'].tolist()
        
        return list(zip(recommended_movies, similarity_scores))
    
    
    def save_model(self, path: str = 'models/collaborative_model.pkl'):
        model_data = {
            'model': self.model,
            'movie_features': self.movie_features,
            'movies': self.movies,
            'indices': self.indices,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    
    @classmethod
    def load_model(cls, path: str = 'models/collaborative_model.pkl'):
        model_data = joblib.load(path)
        
        recommender = cls(
            n_neighbors=model_data['n_neighbors'],
            metric=model_data['metric']
        )
        recommender.model = model_data['model']
        recommender.movie_features = model_data['movie_features']
        recommender.movies = model_data['movies']
        recommender.indices = model_data['indices']
        
        print(f"Model loaded from {path}")
        
        return recommender