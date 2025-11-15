import pandas as pd
import numpy as np
import json
import ast
from typing import List


def parse_json_column(text):
    try:
        return ast.literal_eval(text)
    except:
        return []


def extract_names(json_list: List[dict], key: str = 'name', limit: int = 3) -> List[str]:
    if isinstance(json_list, list):
        return [item[key] for item in json_list[:limit] if key in item]
    return []


def extract_director(crew_list: List[dict]) -> List[str]:
    if isinstance(crew_list, list):
        directors = [member['name'] for member in crew_list if member.get('job') == 'Director']
        return directors[:1]  # Usually one director
    return []


def clean_text(text: str) -> str:
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Select relevant columns
    columns_to_keep = ['id', 'title', 'overview', 'genres', 'keywords', 
                       'cast', 'crew', 'vote_average', 'vote_count', 
                       'popularity', 'release_date']
    
    df = df[columns_to_keep]
    
    # Handle missing values
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    df['cast'] = df['cast'].fillna('[]')
    df['crew'] = df['crew'].fillna('[]')
    
    # Parse JSON columns
    print("Parsing JSON columns...")
    df['genres'] = df['genres'].apply(parse_json_column)
    df['keywords'] = df['keywords'].apply(parse_json_column)
    df['cast'] = df['cast'].apply(parse_json_column)
    df['crew'] = df['crew'].apply(parse_json_column)
    
    # Extract features
    print("Extracting features...")
    df['genres'] = df['genres'].apply(lambda x: extract_names(x, 'name', limit=5))
    df['keywords'] = df['keywords'].apply(lambda x: extract_names(x, 'name', limit=5))
    df['cast'] = df['cast'].apply(lambda x: extract_names(x, 'name', limit=3))
    df['director'] = df['crew'].apply(extract_director)
    
    # Clean text (remove spaces for better matching)
    print("Cleaning text...")
    df['genres'] = df['genres'].apply(lambda x: [clean_text(i) for i in x])
    df['keywords'] = df['keywords'].apply(lambda x: [clean_text(i) for i in x])
    df['cast'] = df['cast'].apply(lambda x: [clean_text(i) for i in x])
    df['director'] = df['director'].apply(lambda x: [clean_text(i) for i in x])
    
    # Filter out movies with low vote counts for quality
    df = df[df['vote_count'] >= 50].reset_index(drop=True)
    
    print(f"Preprocessing complete. Final shape: {df.shape}")
    
    return df


def create_soup(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating feature soup...")
    
    df = df.copy()
    
    # Convert lists to strings
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
    df['keywords_str'] = df['keywords'].apply(lambda x: ' '.join(x))
    df['cast_str'] = df['cast'].apply(lambda x: ' '.join(x))
    df['director_str'] = df['director'].apply(lambda x: ' '.join(x))
    
    # Create soup: combine all text features
    # Give more weight to important features by repeating them
    df['soup'] = (
        df['overview'] + ' ' +
        df['genres_str'] + ' ' + df['genres_str'] + ' ' +  # Double weight
        df['keywords_str'] + ' ' +
        df['cast_str'] + ' ' +
        df['director_str'] + ' ' + df['director_str']  # Double weight
    )
    
    # Clean soup
    df['soup'] = df['soup'].str.lower().str.strip()
    
    print("Feature soup created successfully")
    
    return df