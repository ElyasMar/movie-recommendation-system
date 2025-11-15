import pytest
import sys
import os
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'version' in data


def test_list_movies(client):
    """Test listing movies endpoint."""
    response = client.get('/api/movies?limit=10')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'movies' in data
    assert 'total' in data
    assert isinstance(data['movies'], list)


def test_search_movies(client):
    """Test search endpoint."""
    response = client.get('/api/search?q=dark&limit=5')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'results' in data
    assert isinstance(data['results'], list)


def test_recommend_missing_title(client):
    """Test recommendation without title."""
    response = client.post(
        '/api/recommend',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_recommend_invalid_method(client):
    """Test recommendation with invalid method."""
    response = client.post(
        '/api/recommend',
        data=json.dumps({
            'title': 'The Dark Knight',
            'method': 'invalid_method'
        }),
        content_type='application/json'
    )
    assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])