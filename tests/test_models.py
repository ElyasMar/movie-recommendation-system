import pytest
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender


# Sample test data
@pytest.fixture
def sample_data():
    data = {
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'overview': [
            'Action movie about heroes',
            'Action movie about superheroes',
            'Romantic comedy about love',
            'Romantic drama about relationships',
            'Sci-fi adventure in space'
        ],
        'genres': [
            ['action', 'adventure'],
            ['action', 'scifi'],
            ['romance', 'comedy'],
            ['romance', 'drama'],
            ['scifi', 'adventure']
        ],
        'keywords': [
            ['hero', 'fight'],
            ['superhero', 'powers'],
            ['love', 'funny'],
            ['love', 'sad'],
            ['space', 'aliens']
        ],
        'cast': [
            ['Actor1', 'Actor2'],
            ['Actor1', 'Actor3'],
            ['Actor4', 'Actor5'],
            ['Actor4', 'Actor6'],
            ['Actor7', 'Actor8']
        ],
        'director': [
            ['Director1'],
            ['Director1'],
            ['Director2'],
            ['Director2'],
            ['Director3']
        ],
        'vote_average': [8.5, 8.0, 7.5, 7.0, 8.5],
        'vote_count': [1000, 950, 800, 750, 1100],
        'popularity': [100.0, 95.0, 80.0, 75.0, 105.0],
        'soup': [
            'action adventure hero fight Actor1 Actor2 Director1',
            'action scifi superhero powers Actor1 Actor3 Director1',
            'romance comedy love funny Actor4 Actor5 Director2',
            'romance drama love sad Actor4 Actor6 Director2',
            'scifi adventure space aliens Actor7 Actor8 Director3'
        ]
    }
    
    return pd.DataFrame(data)


class TestContentBasedRecommender:
    """Tests for ContentBasedRecommender."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ContentBasedRecommender()
        assert model.tfidf is None
        assert model.tfidf_matrix is None
        assert model.cosine_sim is None
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        model = ContentBasedRecommender()
        model.fit(sample_data, soup_column='soup')
        
        assert model.tfidf is not None
        assert model.tfidf_matrix is not None
        assert model.cosine_sim is not None
        assert len(model.indices) == len(sample_data)
    
    def test_recommendations(self, sample_data):
        """Test getting recommendations."""
        model = ContentBasedRecommender()
        model.fit(sample_data, soup_column='soup')
        
        # Get recommendations for an action movie
        recs = model.get_recommendations('Movie A', n=2)
        
        assert len(recs) == 2
        assert all(isinstance(r[0], str) for r in recs)  # Titles are strings
        assert all(isinstance(r[1], float) for r in recs)  # Scores are floats
        
        # Should recommend similar action movie
        rec_titles = [r[0] for r in recs]
        assert 'Movie B' in rec_titles or 'Movie E' in rec_titles
    
    def test_invalid_movie(self, sample_data):
        """Test with invalid movie title."""
        model = ContentBasedRecommender()
        model.fit(sample_data, soup_column='soup')
        
        with pytest.raises(ValueError):
            model.get_recommendations('Nonexistent Movie', n=5)
    
    def test_get_movie_details(self, sample_data):
        """Test getting movie details."""
        model = ContentBasedRecommender()
        model.fit(sample_data, soup_column='soup')
        
        details = model.get_movie_details('Movie A')
        
        assert details is not None
        assert details['title'] == 'Movie A'
        assert 'overview' in details
        assert 'genres' in details
        assert details['vote_average'] == 8.5


class TestCollaborativeRecommender:
    """Tests for CollaborativeRecommender."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = CollaborativeRecommender(n_neighbors=5, metric='cosine')
        assert model.n_neighbors == 5
        assert model.metric == 'cosine'
        assert model.model is None
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        model = CollaborativeRecommender(n_neighbors=3)
        model.fit(sample_data)
        
        assert model.model is not None
        assert model.movie_features is not None
        assert len(model.indices) == len(sample_data)
    
    def test_recommendations(self, sample_data):
        """Test getting recommendations."""
        model = CollaborativeRecommender(n_neighbors=3)
        model.fit(sample_data)
        
        recs = model.get_recommendations('Movie A', n=2)
        
        assert len(recs) == 2
        assert all(isinstance(r[0], str) for r in recs)
        assert all(isinstance(r[1], float) for r in recs)
    
    def test_invalid_movie(self, sample_data):
        """Test with invalid movie title."""
        model = CollaborativeRecommender(n_neighbors=3)
        model.fit(sample_data)
        
        with pytest.raises(ValueError):
            model.get_recommendations('Nonexistent Movie', n=2)


def test_model_comparison(sample_data):
    """Test that both models can work on same data."""
    content_model = ContentBasedRecommender()
    content_model.fit(sample_data, soup_column='soup')
    
    collab_model = CollaborativeRecommender(n_neighbors=3)
    collab_model.fit(sample_data)
    
    # Both should give recommendations
    content_recs = content_model.get_recommendations('Movie A', n=2)
    collab_recs = collab_model.get_recommendations('Movie A', n=2)
    
    assert len(content_recs) == 2
    assert len(collab_recs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])