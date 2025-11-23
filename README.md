# movie-recommendation-system

A machine learning based movie recommendation system using collaborative filtering and content based filtering techniques.

## Dataset

- **Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Files:** `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`

## Tech Stack

- **Python 3.9+**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **API:** Flask
- **UI:** Streamlit
- **Deployment:** Docker

## Installation and Setup

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

## Step 1: Clone Repository

git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

## Step 2: Download Dataset

1. Visit Kaggle TMDB Dataset
2. Download both CSV files:

- tmdb_5000_movies.csv
- tmdb_5000_credits.csv


3. Place files in data/raw/ directory
mkdir -p data/raw

## Step 3:Environment setup

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list

## Step 4: Train Models

# Run training pipeline
python src/train.py

# Expected output:
# - Models saved to models/
# - Processed data saved to data/processed/
# - Training takes approximately 2-3 minutes

## Step 5: Verify Installation

# Test model loading
python -c "from src.models.content_based import ContentBasedRecommender; \
           model = ContentBasedRecommender.load_model('models/content_based_model.pkl'); \
           print('Model loaded successfully')"

# Run tests
pytest tests/ -v

## Reproducibility
### Complete Reproduction Steps
- This project is fully reproducible. Follow these steps to reproduce all results:

## 1. Environment Setup
# Clone repository
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install exact dependency versions
pip install -r requirements.txt

## 2. Data Acquisition
# Create data directory
mkdir -p data/raw

# Download data from Kaggle
# Manual download required from:
# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

# Place files in data/raw/:
# - tmdb_5000_movies.csv
# - tmdb_5000_credits.csv

## 3. Training Pipeline
# Run complete training pipeline
python src/train.py

# This will:
# - Load raw data from data/raw/
# - Preprocess and clean data
# - Engineer features
# - Train Content-Based model
# - Train Collaborative model
# - Evaluate both models
# - Save models to models/
# - Save processed data to data/processed/
# - Generate evaluation report

## 4. Model Evaluation
# Run comprehensive model comparison
python compare_models.py

# Results saved to:
# - docs/ACCURACY_RESULTS.md
# - docs/accuracy_metrics.csv

## 5. Exploratory Data Analysis
# Launch Jupyter notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# Execute all cells to reproduce EDA
# Reproducibility Guarantees

# - Fixed Random Seeds: Not applicable (deterministic algorithms used)
# - Dependency Versions: Exact versions specified in requirements.txt
# - Dataset: Publicly available on Kaggle
# - Training Process: Deterministic (no random initialization)
# - Expected Results: Content-Based 83.44%, Collaborative 72.31%

## Validation
- To validate reproduction:


# Compare model accuracy
python -c "
from src.models.content_based import ContentBasedRecommender
model = ContentBasedRecommender.load_model('models/content_based_model.pkl')
recs = model.get_recommendations('The Dark Knight', n=5)
print('Top recommendation:', recs[0][0])
print('Expected: The Dark Knight Rises')
"
```

Expected output:
```
Top recommendation: The Dark Knight Rises
Expected: The Dark Knight Rises
```

---

## Model Deployment

### Flask REST API

The trained models are deployed as a REST API using Flask, providing programmatic access to recommendations.

#### API Architecture

- **Framework**: Flask 3.0.0
- **CORS**: Enabled for cross-origin requests
- **Model Loading**: Models loaded at startup and cached in memory
- **Response Format**: JSON
- **Error Handling**: Comprehensive error responses with appropriate HTTP status codes

#### Available Endpoints

**1. Health Check**
```
GET /api/health
```
Returns API status and version information.

**2. List Movies**
```
GET /api/movies?limit=50&offset=0&search=query
```
Lists available movies with pagination and search.

**3. Movie Details**
```
GET /api/movies/<title>
```
Returns detailed information about a specific movie.

**4. Get Recommendations**
```
POST /api/recommend
Content-Type: application/json

{
  "title": "The Dark Knight",
  "method": "content",
  "n": 10
}
```
Returns movie recommendations. Methods: content, collaborative, hybrid.

**5. Search Movies**
```
GET /api/search?q=batman&limit=10

## Starting the API Server

# Activate virtual environment
source venv/bin/activate

# Start Flask API
python src/api/app.py

# Server starts on http://0.0.0.0:5000

## Testing the API
# Health check
curl http://localhost:5000/api/health

# Get recommendations
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "method": "content", "n": 5}'

# Search movies
curl "http://localhost:5000/api/search?q=matrix&limit=3"

## Starting the Web Interface
# Activate virtual environment
source venv/bin/activate

# Start Streamlit
streamlit run src/streamlit_app.py

# Opens automatically at http://localhost:8501

## Running Both Services
# Terminal 1: API
python src/api/app.py

# Terminal 2: Web UI
streamlit run src/streamlit_app.py

## Building the Docker Image
# Build image
docker build -t movie-recommender .

# Build takes approximately 10-15 minutes
# Image size: ~1.5 GB

## Running the Container
# Run container
docker run -d \
  -p 8501:8501 \
  -p 5000:5000 \
  --name movie-rec \
  --restart unless-stopped \
  movie-recommender

# Check container status
docker ps

# View logs
docker logs -f movie-rec

## Container Management
# Stop container
docker stop movie-rec

# Start container
docker start movie-rec

# Restart container
docker restart movie-rec

# Remove container
docker stop movie-rec
docker rm movie-rec

# Remove image
docker rmi movie-recommender

## Docker Compose
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

## Features

- **Content-Based Filtering:** Recommends movies based on genres, keywords, cast, and crew
- **Collaborative Filtering:** User based recommendations
- **Hybrid Approach:** Combines both methods
- **REST API:** Flask-based API for predictions
- **Interactive UI:** Streamlit dashboard
- **Dockerized:** Ready for deployment

## Model Performance

The system compares three approaches:

1. **Content-Based Filtering** (TF-IDF + Cosine Similarity)  
   - Uses movie overview, genres, keywords, cast, director  
   - Strengths: High genre match, explainable recommendations  

2. **Collaborative Filtering** (KNN)  
   - Uses popularity, vote average, vote count  
   - Strengths: Very fast, good rating match  

3. **Hybrid** (60% Content-Based + 40% Collaborative)  
   - Strengths: Balanced recommendations  

### Key Metrics

| Model | Accuracy | Genre Match | Rating Match | Precision@10 | Speed |
|-------|----------|-------------|--------------|--------------|-------|
| Content-Based | **83.44%** | **96.11%** | 66.11% | **86.11%** | 9.6ms |
| Collaborative | 72.31% | 52.78% | **90%** | 72.78% | **2.33ms** |
| Hybrid | 72.31% | 52.78% | 90% | 72.78% | 11.01ms |

### Recommended Model: Content-Based
- Best overall accuracy and genre matching  
- Fast enough for real-time use (9.6ms)  
- Precision@10: 86.11%  

**Notes:**  
- Collaborative is faster (2.33ms) but less accurate overall  
- Hybrid balances both approaches but doesn’t outperform individually



## Author

Ilias Cherkaoui