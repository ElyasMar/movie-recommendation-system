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

Models compared:
- Cosine Similarity (Content-Based)
- Nearest Neighbors (Collaborative)
- TF-IDF Vectorization

## 👥 Author

Ilias Cherkaoui