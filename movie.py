import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

# Set Streamlit page configuration
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# Apply custom styling for light theme
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            color: black;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
        }
        .stDataFrame {
            border: 2px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Upload CSV File
st.title("🎬 Movie Recommendation System")
st.markdown("##### Upload your dataset to get started!")
uploaded_file = st.file_uploader("📂 Upload a Movie Dataset (CSV)", type=["csv"])

# Load Dataset Function
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        
        # Standardize column names
        column_mapping = {"Title": "title", "Genre": "genres"}
        df.rename(columns=column_mapping, inplace=True)
        
        # Required columns
        required_columns = ["title", "overview", "genres", "popularity", "director", "rating", "reviews"]

        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col == "overview":
                    df[col] = "No overview available"
                elif col == "popularity":
                    df[col] = np.random.randint(1, 100, size=len(df))  # Assign random popularity
                elif col == "rating":
                    df[col] = np.random.uniform(1, 10, size=len(df)).round(1)  # Assign random ratings
                elif col == "reviews":
                    df[col] = np.random.randint(1, 500, size=len(df))  # Assign random review counts
                else:
                    df[col] = "Unknown"  # Fill director, genres, etc.
        
        return df
    except Exception as e:
        st.error(f"⚠ Error loading dataset: {e}")
        return None

# Load movies if a file is uploaded
if uploaded_file is not None:
    movies = load_data(uploaded_file)
else:
    movies = None

# Display dataset
if movies is not None:
    st.markdown("### 📊 Sample Data")
    st.dataframe(movies.head())
    
    # Search Bar for Movies
    st.markdown("### 🔍 Search for a Movie")
    search_query = st.text_input("Enter movie name:")
    if search_query:
        results = movies[movies["title"].str.contains(search_query, case=False, na=False)]
        st.dataframe(results)
    
    # Movie Popularity Analysis
    st.markdown("### 📈 Popularity vs Rating")
    fig = px.scatter(movies, x="popularity", y="rating", size="reviews", color="genres", hover_name="title")
    st.plotly_chart(fig)
    
    # Bar Chart for Top Rated Movies
    st.markdown("### 🎬 Top 10 Rated Movies")
    top_movies = movies.sort_values(by="rating", ascending=False).head(10)
    fig_bar = px.bar(top_movies, x="title", y="rating", color="rating", text="rating")
    st.plotly_chart(fig_bar)
    
    # Suggest Movie Trailer Links
    st.markdown("### 🎥 Watch Trailer")
    movie_choice = st.selectbox("Select a movie to watch its trailer:", movies["title"].unique())
    search_url = f"https://www.youtube.com/results?search_query={movie_choice}+trailer"
    if st.button("Watch Trailer 🎬"):
        webbrowser.open(search_url)