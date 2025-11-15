import pandas as pd
import os
from typing import Tuple

def load_data(data_path: str = 'data/raw/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_file = os.path.join(data_path, 'tmdb_5000_movies.csv')
    credits_file = os.path.join(data_path, 'tmdb_5000_credits.csv')
    
    if not os.path.exists(movies_file):
        raise FileNotFoundError(f"Movies file not found: {movies_file}")
    if not os.path.exists(credits_file):
        raise FileNotFoundError(f"Credits file not found: {credits_file}")
    
    print(f"Loading movies from {movies_file}...")
    movies = pd.read_csv(movies_file)
    
    print(f"Loading credits from {credits_file}...")
    credits = pd.read_csv(credits_file)
    
    print(f"Loaded {len(movies)} movies and {len(credits)} credits records")
    
    return movies, credits


def merge_datasets(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    # Merge on title
    merged = movies.merge(credits, left_on='title', right_on='title', how='left')
    
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Columns: {merged.columns.tolist()}")
    
    return merged


def save_processed_data(df: pd.DataFrame, output_path: str = 'data/processed/movies_processed.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")