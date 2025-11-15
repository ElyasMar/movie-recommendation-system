import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender


# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():
    content_model = ContentBasedRecommender.load_model('models/content_based_model.pkl')
    collab_model = CollaborativeRecommender.load_model('models/collaborative_model.pkl')
    return content_model, collab_model


def main():
    # Title and description
    st.title("Movie Recommendation System")
    st.markdown("""
    Discover your next favorite movie! This system uses machine learning to recommend 
    movies based on content similarity and collaborative filtering.
    """)
    
    # Load models
    try:
        with st.spinner("Loading models..."):
            content_model, collab_model = load_models()
        
        movies_df = content_model.movies
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please run `python src/train.py` first to train the models.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Recommendation method
    method = st.sidebar.selectbox(
        "Recommendation Method",
        ["Content-Based", "Collaborative", "Hybrid"],
        help="Content-Based: Similar genres, cast, keywords\nCollaborative: Similar popularity and ratings\nHybrid: Combination of both"
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Browse Movies", "ℹ️ About"])
    
    # Tab 1: Recommendations
    with tab1:
        st.header("Get Movie Recommendations")
        
        # Movie selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search box
            search_query = st.text_input(
                "Search for a movie:",
                placeholder="Type movie title..."
            )
            
            # Filter movies based on search
            if search_query:
                filtered_movies = movies_df[
                    movies_df['title'].str.contains(search_query, case=False, na=False)
                ]['title'].tolist()
            else:
                # Show popular movies
                filtered_movies = movies_df.nlargest(100, 'popularity')['title'].tolist()
            
            selected_movie = st.selectbox(
                "Select a movie:",
                options=filtered_movies,
                index=0 if filtered_movies else None
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            recommend_button = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
        
        # Show recommendations
        if recommend_button and selected_movie:
            with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
                try:
                    # Get movie details
                    movie_details = content_model.get_movie_details(selected_movie)
                    
                    # Display selected movie info
                    st.subheader(f"Selected Movie: {selected_movie}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rating", f"⭐ {movie_details['vote_average']:.1f}/10")
                    with col2:
                        st.metric("Votes", f"🗳️ {movie_details['vote_count']:,}")
                    with col3:
                        st.metric("Popularity", f"🔥 {movie_details['popularity']:.0f}")
                    
                    st.write("**Genres:**", ", ".join(movie_details['genres']))
                    st.write("**Overview:**", movie_details['overview'])
                    
                    st.divider()
                    
                    # Get recommendations based on method
                    if method == "Content-Based":
                        recommendations = content_model.get_recommendations(
                            selected_movie, n=n_recommendations
                        )
                    elif method == "Collaborative":
                        recommendations = collab_model.get_recommendations(
                            selected_movie, n=n_recommendations
                        )
                    else:  # Hybrid
                        content_recs = content_model.get_recommendations(
                            selected_movie, n=n_recommendations
                        )
                        collab_recs = collab_model.get_recommendations(
                            selected_movie, n=n_recommendations
                        )
                        
                        # Combine scores
                        combined = {}
                        for movie, score in content_recs:
                            combined[movie] = score * 0.6
                        for movie, score in collab_recs:
                            if movie in combined:
                                combined[movie] += score * 0.4
                            else:
                                combined[movie] = score * 0.4
                        
                        recommendations = sorted(
                            combined.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:n_recommendations]
                    
                    # Display recommendations
                    st.subheader(f"🎯 Top {len(recommendations)} Recommendations ({method})")
                    
                    for i, (title, score) in enumerate(recommendations, 1):
                        rec_details = content_model.get_movie_details(title)
                        
                        with st.expander(f"#{i} - {title} (Score: {score:.3f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Overview:**", rec_details['overview'])
                                st.write("**Genres:**", ", ".join(rec_details['genres']))
                            
                            with col2:
                                st.metric("Rating", f"⭐ {rec_details['vote_average']:.1f}")
                                st.metric("Votes", f"{rec_details['vote_count']:,}")
                
                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
    
    # Tab 2: Browse Movies
    with tab2:
        st.header("Browse All Movies")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.5)
        
        with col2:
            min_votes = st.slider("Minimum Votes", 0, 5000, 100, 100)
        
        # Filter dataframe
        filtered_df = movies_df[
            (movies_df['vote_average'] >= min_rating)
            & (movies_df['vote_count'] >= min_votes)
        ].sort_values('popularity', ascending=False)
        
        st.write(f"**Showing {len(filtered_df)} movies**")
        
        # Display dataframe
        display_df = filtered_df[['title', 'vote_average', 'vote_count', 'popularity', 'genres']].head(50)
        
        st.dataframe(
            display_df,
            column_config={
                "title": "Movie Title",
                "vote_average": st.column_config.NumberColumn(
                    "Rating",
                    format="⭐ %.1f"
                ),
                "vote_count": st.column_config.NumberColumn(
                    "Votes",
                    format="%d"
                ),
                "popularity": st.column_config.NumberColumn(
                    "Popularity",
                    format="%.0f"
                ),
                "genres": "Genres"
            },
            hide_index=True,
            use_container_width=True
        )

    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Project Overview
        """)
        
        # Statistics
        st.divider()
        st.subheader("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")
        
        with col2:
            st.metric("Avg Rating", f"{movies_df['vote_average'].mean():.2f}")
        
        with col3:
            st.metric("Total Votes", f"{movies_df['vote_count'].sum():,}")
        
        with col4:
            unique_genres = set()
            for genres in movies_df['genres']:
                unique_genres.update(genres)
            st.metric("Unique Genres", len(unique_genres))


if __name__ == "__main__":
    main()