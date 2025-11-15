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

The movie recommendation system uses three approaches: **Content-Based**, **Collaborative**, and **Hybrid** recommenders.  

- **Content-Based:** Uses TF-IDF vectorization of movie attributes and cosine similarity. Provides diverse recommendations across genres.  
- **Collaborative:** Uses Nearest Neighbors (KNN) on normalized numerical features like popularity, vote average, and vote count. Produces highly similar recommendations but less genre diversity.  
- **Hybrid:** Combines Content-Based and Collaborative recommendations, achieving a balance of similarity and diversity.  

**Evaluation Results:**

- Success Rate: 100% for all models  
- Average Response Time: Collaborative (fastest, 7.02 ms), Content-Based (11.11 ms), Hybrid (16.29 ms)  
- Average Similarity Score: Collaborative (0.998), Content-Based (0.237), Hybrid (0.402)  
- Genre Diversity: Hybrid (highest, 0.386), Content-Based (0.263), Collaborative (0.0)  

**Conclusion:**  
The **Hybrid recommender** is the best overall, combining the strengths of both approaches to provide relevant and diverse recommendations.


## 👥 Author

Ilias Cherkaoui