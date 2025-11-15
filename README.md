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

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

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