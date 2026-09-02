import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pyarrow as pa
from fastparquet import ParquetFile
import matplotlib.dates as mpl_dates

# Set page config
st.set_page_config(
    page_title="Movie Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up the app title and description
st.title("🎬 Movie Analysis Dashboard")
st.markdown("""
Visualizing movie ratings and analysis from your Trakt.tv data.
""")

# Add the project root to sys.path
sys.path.append(str(Path().resolve().parent))

# Set folder path
root_path = Path().resolve().parent
file_path = root_path / "data" / "processed"

@st.cache_data
def load_data():
    # Get all PARQ files
    parq_files = list(file_path.glob('*.parq*'))
    dfs = {file.stem: pd.read_parquet(file) for file in parq_files}
    
    # Create DataFrames
    df = pd.DataFrame(dfs['cleaned_enriched_movies'])
    df_genres = pd.DataFrame(dfs['cleaned_enriched_movies_genres'])
    df_directors = pd.DataFrame(dfs['cleaned_enriched_movies_directors'])
    df_actors = pd.DataFrame(dfs['cleaned_enriched_movies_actors'])
    
    # Convert date columns to datetime
    date_columns = ['Date Rated', 'Release Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df, df_genres, df_directors, df_actors

# Load the data
df, df_genres, df_directors, df_actors = load_data()

# Create viewer selector in sidebar
viewer = st.sidebar.radio("Select Viewer", ['All', 'nowell', 'taylor'])

# Filter data based on viewer selection
if viewer != 'All':
    df = df[df['Viewer'] == viewer]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Genres", "Directors & Actors", "Temporal Analysis"])

with tab1:
    st.header("Overview of Ratings")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Average ratings
        st.subheader("Average Ratings")
        avg_ratings = df.groupby('Viewer').agg({
            'User Rating': 'mean',
            'IMDb Rating': 'mean'
        }).round(2)
        st.dataframe(avg_ratings)
    
    with col2:
        # Rating distribution
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='User Rating', bins=10, kde=True, ax=ax)
        st.pyplot(fig)
    
    # Time series of ratings
    st.subheader("Ratings Over Time")
    time_group = st.selectbox(
        "Group by:",
        ["Year", "Quarter", "Month"]
    )
    
    if time_group == "Year":
        freq = 'YE'
    elif time_group == "Quarter":
        freq = 'QE'
    else:
        freq = 'ME'
    
    time_series = df.groupby([pd.Grouper(key='Date Rated', freq=freq)])[['User Rating', 'IMDb Rating']].mean()
    time_series.index = time_series.index.strftime('%Y-%m')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time_series.plot(ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Genre Analysis")
    
    # Most common genres
    st.subheader("Most Common Genres")
    min_count = st.slider("Minimum number of ratings per genre", 1, 50, 5)
    
    genre_counts = df_genres['Genres'].value_counts()
    valid_genres = genre_counts[genre_counts >= min_count].index
    filtered_df = df_genres[df_genres['Genres'].isin(valid_genres)]
    
    genre_ratings = filtered_df.groupby(['Genres','Viewer']).agg({
        'User Rating':'mean',
        'IMDb Rating':'mean',
        'Genres':'count'
    }).rename(columns={'Genres': 'Count'}).sort_values(by='Count', ascending=False)
    
    genre_ratings = genre_ratings.reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=genre_ratings.head(20),
        y='Genres', x='Count', hue='Viewer',
        ax=ax
    )
    plt.title('Most Common Genres')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Genre ratings comparison
    st.subheader("Genre Ratings Comparison")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=genre_ratings.head(20),
        y='Genres', x='User Rating', hue='Viewer',
        ax=ax
    )
    plt.title('Average Ratings by Genre')
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.header("Directors & Actors Analysis")
    
    # Top directors
    st.subheader("Top Directors by Rating")
    min_director_count = st.slider("Minimum number of movies per director", 1, 20, 3)
    
    director_summary = df_directors.groupby(['Directors','Viewer']).agg({
        'User Rating': 'mean',
        'IMDb Rating': 'mean',
        'Directors': 'count'
    }).rename(columns={'Directors': 'Count'}).sort_values(by='User Rating', ascending=False)
    
    director_summary = director_summary[director_summary['Count'] >= min_director_count].reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=director_summary.head(10),
        x='Directors', y='User Rating', hue='Viewer',
        ax=ax
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Top Rated Directors')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top actors
    st.subheader("Top Actors by Rating")
    min_actor_count = st.slider("Minimum number of movies per actor", 1, 20, 3)
    
    actor_summary = df_actors.groupby(['Actors','Viewer']).agg({
        'User Rating': 'mean',
        'IMDb Rating': 'mean',
        'Actors': 'count'
    }).rename(columns={'Actors': 'Count'}).sort_values(by='User Rating', ascending=False)
    
    actor_summary = actor_summary[actor_summary['Count'] >= min_actor_count].reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=actor_summary.head(10),
        x='Actors', y='User Rating', hue='Viewer',
        ax=ax
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Top Rated Actors')
    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    st.header("Temporal Analysis")
    
    # Ratings over time
    st.subheader("Ratings by Release Year")
    
    # Create a slider to select year range
    min_year = int(df['Release Date'].dt.year.min())
    max_year = int(df['Release Date'].dt.year.max())
    year_range = st.slider(
        "Select year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filter data by selected year range
    filtered_df = df[
        (df['Release Date'].dt.year >= year_range[0]) & 
        (df['Release Date'].dt.year <= year_range[1])
    ]
    
    # Plot ratings by release year
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x=filtered_df['Release Date'].dt.year,
        y='User Rating',
        data=filtered_df,
        ax=ax
    )
    plt.xticks(rotation=45)
    plt.title('User Ratings by Release Year')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Monthly/Seasonal patterns
    st.subheader("Seasonal Patterns")
    
    # Extract month and season
    filtered_df['Month'] = filtered_df['Release Date'].dt.month
    filtered_df['Season'] = filtered_df['Month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else 
                 'Spring' if x in [3, 4, 5] else 
                 'Summer' if x in [6, 7, 8] else 'Fall'
    )
    
    # Plot ratings by season
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='Season',
        y='User Rating',
        data=filtered_df,
        order=['Winter', 'Spring', 'Summer', 'Fall'],
        ax=ax
    )
    plt.title('User Ratings by Release Season')
    st.pyplot(fig)

# Add some styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)