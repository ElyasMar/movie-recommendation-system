# Movie Recommendation System

A **machine learning–based movie recommendation system** that combines **content-based filtering**, **collaborative filtering**, and a **hybrid approach**. The project provides a complete pipeline from data preprocessing and model training to deployment via a **Flask REST API**, **Streamlit web interface**, and **Docker**.

---

## Dataset

* **Source:** TMDB 5000 Movie Dataset (Kaggle)
* **Files used:**

  * `tmdb_5000_movies.csv`
  * `tmdb_5000_credits.csv`

---

## Tech Stack

* **Python:** 3.9+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **API:** Flask
* **Web UI:** Streamlit
* **Deployment:** Docker, Docker Compose

---

## Installation and Setup

### Prerequisites

* Python 3.9 or higher
* Git
* Docker (optional, for containerized deployment)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

---

### Step 2: Download the Dataset

1. Visit the TMDB 5000 Movie Dataset on Kaggle

2. Download the following files:

   * `tmdb_5000_movies.csv`
   * `tmdb_5000_credits.csv`

3. Create the data directory and place the files inside:

```bash
mkdir -p data/raw
```

```
data/raw/
 ├── tmdb_5000_movies.csv
 └── tmdb_5000_credits.csv
```

---

### Step 3: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4: Train the Models

```bash
python src/train.py
```

This step will:

* Load raw data from `data/raw/`
* Clean and preprocess the data
* Engineer features
* Train content-based and collaborative models
* Evaluate models
* Save trained models to `models/`
* Save processed data to `data/processed/`

> Training time: approximately **2–3 minutes**

---

### Step 5: Verify Installation

```bash
python -c "from src.models.content_based import ContentBasedRecommender; model = ContentBasedRecommender.load_model('models/content_based_model.pkl'); print('Model loaded successfully')"
```

Run tests if available:

```bash
pytest tests/ -v
```

---

## Reproducibility

This project is **fully reproducible**.

### Complete Reproduction Steps

#### 1. Environment Setup

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Data Acquisition

```bash
mkdir -p data/raw
```

Manually download the dataset from Kaggle and place the CSV files into `data/raw/`.

#### 3. Training Pipeline

```bash
python src/train.py
```

#### 4. Model Evaluation

```bash
python compare_models.py
```

Results will be saved to:

* `docs/ACCURACY_RESULTS.md`
* `docs/accuracy_metrics.csv`

#### 5. Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

### Reproducibility Guarantees

* Dataset: Publicly available (Kaggle)
* Deterministic training process
* Dependency versions fixed in `requirements.txt`
* Expected accuracy results:

  * Content-Based: **83.44%**
  * Collaborative: **72.31%**

---

## Validation Example

```bash
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

The trained models are exposed via a **Flask-based REST API**.

#### Available Endpoints

* **Health Check**

  ```http
  GET /api/health
  ```

* **List Movies**

  ```http
  GET /api/movies?limit=50&offset=0&search=query
  ```

* **Movie Details**

  ```http
  GET /api/movies/<title>
  ```

* **Get Recommendations**

  ```http
  POST /api/recommend
  Content-Type: application/json

  {
    "title": "The Dark Knight",
    "method": "content",
    "n": 10
  }
  ```

* **Search Movies**

  ```http
  GET /api/search?q=batman&limit=10
  ```

---

### Starting the API Server

```bash
source venv/bin/activate
python src/api/app.py
```

API runs at: **[http://localhost:5000](http://localhost:5000)**

---

## Web Interface (Streamlit)

```bash
source venv/bin/activate
streamlit run src/streamlit_app.py
```

UI available at: **[http://localhost:8501](http://localhost:8501)**

---

## Docker Deployment

### Build Image

```bash
docker build -t movie-recommender .
```

### Run Container

```bash
docker run -d \
  -p 8501:8501 \
  -p 5000:5000 \
  --name movie-rec \
  --restart unless-stopped \
  movie-recommender
```

---

### Docker Compose

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Features

* Content-Based movie recommendations
* Collaborative filtering recommendations
* Hybrid recommendation strategy
* REST API for programmatic access
* Interactive Streamlit dashboard
* Fully Dockerized deployment

---

## Model Performance

| Model         | Accuracy   | Genre Match | Rating Match | Precision@10 | Speed       |
| ------------- | ---------- | ----------- | ------------ | ------------ | ----------- |
| Content-Based | **83.44%** | **96.11%**  | 66.11%       | **86.11%**   | 9.6 ms      |
| Collaborative | 72.31%     | 52.78%      | **90%**      | 72.78%       | **2.33 ms** |
| Hybrid        | 72.31%     | 52.78%      | 90%          | 72.78%       | 11.01 ms    |

### Recommended Model

**Content-Based Filtering** offers the best balance of accuracy, explainability, and real-time performance.

---

## Author

**Ilias Cherkaoui**
