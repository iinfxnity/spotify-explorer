import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import kagglehub
import os

# Page config
st.set_page_config(
    page_title="Spotify Music Explorer",
    page_icon="🎵",
    layout="wide"
)

# CSS Styling
st.markdown("""
    <style>
    /* Main background - Spotify black */
    .stApp {
        background-color: #121212;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-family: 'Circular', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .subtitle {
        text-align: center;
        color: #b3b3b3;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics styling - Spotify cards */
    [data-testid="stMetricValue"] {
        color: #1DB954;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #282828 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #282828;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #181818;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #b3b3b3;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #282828;
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #000000 !important;
    }
    
    /* Buttons - Spotify green */
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 500px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background-color: #1ed760;
        transform: scale(1.05);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #1DB954;
    }
    
    /* Selectbox and multiselect */
    .stSelectbox label, .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #181818;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #282828;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #181818;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #181818;
        color: #ffffff;
        border-left: 4px solid #1DB954;
        border-radius: 8px;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #b3b3b3;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        border-radius: 12px;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1 {
        color: #1DB954 !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #1DB954;
        color: #000000;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #181818;
        border: 2px dashed #1DB954;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1ed760;
        background-color: #1a1a1a;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1DB954;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1ed760;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Spotify Music Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover insights from thousands of Spotify tracks</p>', unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.radio(
    "Choose dataset source:",
    ["Load from Kaggle", "Upload CSV file"]
)

df = None

if dataset_option == "Load from Kaggle":
    # Kaggle dataset options
    kaggle_datasets = {
        "Spotify Tracks Dataset (114k+ tracks, recommended)": "maharshipandya/-spotify-tracks-dataset",
        "Most Streamed Spotify Songs 2024": "nelgiriyewithana/most-streamed-spotify-songs-2024",
        "Spotify 1 Million Tracks": "amitanshjoshi/spotify-1million-tracks"
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Select Kaggle dataset:",
        list(kaggle_datasets.keys())
    )
    
    load_button = st.sidebar.button("📥 Load Dataset from Kaggle", type="primary")
    
    if load_button:
        with st.spinner(f"Downloading dataset from Kaggle... This may take a minute on first run."):
            try:
                # Download dataset
                dataset_path = kagglehub.dataset_download(kaggle_datasets[selected_dataset])
                
                # Find CSV file in the downloaded path
                csv_files = []
                for root, dirs, files in os.walk(dataset_path):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                
                if csv_files:
                    # Load the first CSV file found
                    df = pd.read_csv(csv_files[0])
                    st.sidebar.success(f"✅ Dataset loaded! {len(df):,} tracks found")
                    st.session_state['df'] = df
                else:
                    st.sidebar.error("No CSV files found in the dataset")
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
                st.sidebar.info("💡 Make sure you have kagglehub installed: `pip install kagglehub`")
    
    # Check if dataset is already loaded in session
    if 'df' in st.session_state and df is None:
        df = st.session_state['df']

else:  # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Upload your Spotify dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df

# File uploader (keeping for backwards compatibility, but moved to sidebar)

if df is not None:
    # Show dataset info
    with st.expander("Dataset Preview"):
        st.write(f"**Total Tracks:** {len(df):,}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        st.dataframe(df.head(10))
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Detect available columns (different datasets have different column names)
    # Common variations for genre/track_genre/artists_genres
    genre_col = None
    for col in ['track_genre', 'genre', 'artists_genres', 'genres']:
        if col in df.columns:
            genre_col = col
            break
    
    # Year filter (only if available)
    year_col = None
    for col in ['year', 'release_year', 'album_release_date']:
        if col in df.columns:
            year_col = col
            break
    
    # Genre filter
    if genre_col and df[genre_col].notna().any():
        unique_genres = sorted(df[genre_col].dropna().unique())
        if len(unique_genres) > 0:
            selected_genres = st.sidebar.multiselect(
                "Select Genres",
                options=unique_genres,
                default=unique_genres[:5] if len(unique_genres) >= 5 else unique_genres
            )
            if selected_genres:
                df = df[df[genre_col].isin(selected_genres)]
    
    # Year filter
    if year_col:
        try:
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            valid_years = df[year_col].dropna()
            if len(valid_years) > 0:
                min_year = int(valid_years.min())
                max_year = int(valid_years.max())
                year_range = st.sidebar.slider(
                    "Release Year Range",
                    min_year, max_year,
                    (min_year, max_year)
                )
                df = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]
        except:
            pass
    
    # Audio features filters (only if available)
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']
    available_features = [f for f in audio_features if f in df.columns]
    
    if available_features:
        st.sidebar.subheader("🎚️ Audio Features")
        for feature in available_features[:3]:  # Show top 3 features
            if df[feature].notna().any():
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                feature_range = st.sidebar.slider(
                    f"{feature.capitalize()}",
                    min_val, max_val,
                    (min_val, max_val),
                    key=feature
                )
                df = df[(df[feature] >= feature_range[0]) & (df[feature] <= feature_range[1])]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📈 **{len(df):,}** tracks match your filters")
    
    # Main content
    if len(df) == 0:
        st.warning("⚠️ No tracks match your filters. Try adjusting them!")
    else:
        # Summary Statistics
        st.markdown("## Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tracks", f"{len(df):,}")
        
        with col2:
            if 'popularity' in df.columns:
                avg_pop = df['popularity'].mean()
                st.metric("Avg Popularity", f"{avg_pop:.1f}")
            elif 'streams' in df.columns:
                total_streams = df['streams'].sum()
                st.metric("Total Streams", f"{total_streams:,.0f}")
        
        with col3:
            if 'duration_ms' in df.columns:
                avg_duration = df['duration_ms'].mean() / 60000
                st.metric("Avg Duration", f"{avg_duration:.1f} min")
            elif 'tempo' in df.columns:
                avg_tempo = df['tempo'].mean()
                st.metric("Avg Tempo", f"{avg_tempo:.0f} BPM")
        
        with col4:
            if 'energy' in df.columns:
                avg_energy = df['energy'].mean()
                st.metric("Avg Energy", f"{avg_energy:.2f}")
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Trends", "ML Clustering"])
        
        with tab1:
            st.markdown("### Distribution of Audio Features")
            
            if available_features:
                col1, col2 = st.columns(2)
                
                with col1:
                    feature1 = st.selectbox("Select feature for histogram", available_features, key='hist1')
                    fig1 = px.histogram(df, x=feature1, nbins=50, 
                                       title=f"Distribution of {feature1.capitalize()}",
                                       color_discrete_sequence=['#1DB954'])
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    if 'popularity' in df.columns:
                        fig2 = px.histogram(df, x='popularity', nbins=50,
                                           title="Distribution of Popularity",
                                           color_discrete_sequence=['#1ed760'])
                        fig2.update_layout(
                            showlegend=False,
                            plot_bgcolor='#181818',
                            paper_bgcolor='#181818',
                            font=dict(color='#ffffff'),
                            title_font=dict(size=20, color='#ffffff')
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    elif len(available_features) > 1:
                        feature2 = st.selectbox("Select second feature", 
                                               [f for f in available_features if f != feature1], 
                                               key='hist2')
                        fig2 = px.histogram(df, x=feature2, nbins=50,
                                           title=f"Distribution of {feature2.capitalize()}",
                                           color_discrete_sequence=['#1ed760'])
                        fig2.update_layout(
                            showlegend=False,
                            plot_bgcolor='#181818',
                            paper_bgcolor='#181818',
                            font=dict(color='#ffffff'),
                            title_font=dict(size=20, color='#ffffff')
                        )
                        st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.markdown("### Relationships Between Features")
            
            if len(available_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_feature = st.selectbox("X-axis", available_features, key='scatter_x')
                
                with col2:
                    y_feature = st.selectbox("Y-axis", 
                                            [f for f in available_features if f != x_feature],
                                            key='scatter_y')
                
                color_by = None
                if genre_col and genre_col in df.columns:
                    color_by = genre_col
                elif 'popularity' in df.columns:
                    color_by = 'popularity'
                
                fig = px.scatter(df.sample(min(1000, len(df))), 
                               x=x_feature, y=y_feature,
                               color=color_by,
                               title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}",
                               opacity=0.6,
                               hover_data=['track_name'] if 'track_name' in df.columns else None,
                               color_continuous_scale='Viridis')
                fig.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font=dict(color='#ffffff'),
                    title_font=dict(size=20, color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                if len(available_features) >= 3:
                    st.markdown("#### Correlation Heatmap")
                    corr_features = available_features[:6]  # Top 6 features
                    corr_matrix = df[corr_features].corr()
                    
                    fig_corr = px.imshow(corr_matrix, 
                                        text_auto='.2f',
                                        title="Feature Correlations",
                                        color_continuous_scale='RdBu_r',
                                        aspect='auto')
                    fig_corr.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font=dict(color='#ffffff'),
                        title_font=dict(size=20, color='#ffffff')
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.markdown("### Trends Over Time")
            
            if year_col and year_col in df.columns:
                trend_feature = st.selectbox(
                    "Select feature to analyze over time",
                    available_features if available_features else ['popularity'],
                    key='trend'
                )
                
                yearly_avg = df.groupby(year_col)[trend_feature].mean().reset_index()
                yearly_avg = yearly_avg.sort_values(year_col)
                
                fig = px.line(yearly_avg, x=year_col, y=trend_feature,
                            title=f"Average {trend_feature.capitalize()} Over Time",
                            markers=True)
                fig.update_traces(line_color='#1DB954', line_width=3)
                fig.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font=dict(color='#ffffff'),
                    title_font=dict(size=20, color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Genre trends
                if genre_col and genre_col in df.columns:
                    st.markdown("#### Genre Popularity Over Time")
                    top_genres = df[genre_col].value_counts().head(5).index
                    genre_yearly = df[df[genre_col].isin(top_genres)].groupby([year_col, genre_col]).size().reset_index(name='count')
                    
                    fig_genre = px.line(genre_yearly, x=year_col, y='count', color=genre_col,
                                       title="Top 5 Genres Over Time",
                                       markers=True)
                    fig_genre.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font=dict(color='#ffffff'),
                        title_font=dict(size=20, color='#ffffff')
                    )
                    st.plotly_chart(fig_genre, use_container_width=True)
            else:
                st.info("Year information not available in this dataset")
        
        with tab4:
            st.markdown("### 🤖 K-Means Clustering")
            st.write("Discover groups of similar songs based on audio features")
            
            if len(available_features) >= 2:
                # Select features for clustering
                cluster_features = st.multiselect(
                    "Select features for clustering",
                    available_features,
                    default=available_features[:4] if len(available_features) >= 4 else available_features
                )
                
                if len(cluster_features) >= 2:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        n_clusters = st.slider("Number of clusters", 2, 10, 4)
                        run_clustering = st.button("🎯 Run Clustering", type="primary")
                    
                    if run_clustering:
                        with st.spinner("Clustering songs..."):
                            # Prepare data
                            X = df[cluster_features].dropna()
                            df_clean = df.loc[X.index].copy()
                            
                            # Standardize features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Perform clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            df_clean['cluster'] = kmeans.fit_predict(X_scaled)
                            
                            # Visualize clusters
                            if len(cluster_features) >= 2:
                                fig = px.scatter(df_clean.sample(min(1000, len(df_clean))),
                                               x=cluster_features[0],
                                               y=cluster_features[1],
                                               color='cluster',
                                               title=f"Clusters based on {', '.join(cluster_features)}",
                                               hover_data=['track_name'] if 'track_name' in df_clean.columns else None,
                                               color_continuous_scale='Viridis')
                                fig.update_layout(
                                    plot_bgcolor='#181818',
                                    paper_bgcolor='#181818',
                                    font=dict(color='#ffffff'),
                                    title_font=dict(size=20, color='#ffffff')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Cluster characteristics
                            st.markdown("#### Cluster Characteristics")
                            cluster_stats = df_clean.groupby('cluster')[cluster_features].mean()
                            
                            for i in range(n_clusters):
                                with st.expander(f"📌 Cluster {i} ({len(df_clean[df_clean['cluster']==i])} songs)"):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        # Radar chart for cluster
                                        values = cluster_stats.loc[i].tolist()
                                        values.append(values[0])  # Close the polygon
                                        features_radar = cluster_features + [cluster_features[0]]
                                        
                                        fig_radar = go.Figure()
                                        fig_radar.add_trace(go.Scatterpolar(
                                            r=values,
                                            theta=features_radar,
                                            fill='toself',
                                            name=f'Cluster {i}',
                                            fillcolor='rgba(29, 185, 84, 0.5)',
                                            line=dict(color='#1DB954', width=2)
                                        ))
                                        fig_radar.update_layout(
                                            polar=dict(
                                                radialaxis=dict(
                                                    visible=True, 
                                                    range=[0, 1],
                                                    gridcolor='#282828',
                                                    linecolor='#282828'
                                                ),
                                                bgcolor='#181818',
                                                angularaxis=dict(
                                                    gridcolor='#282828',
                                                    linecolor='#282828'
                                                )
                                            ),
                                            showlegend=False,
                                            height=300,
                                            paper_bgcolor='#181818',
                                            font=dict(color='#ffffff')
                                        )
                                        st.plotly_chart(fig_radar, use_container_width=True)
                                    
                                    with col2:
                                        st.write("**Average values:**")
                                        for feat in cluster_features:
                                            st.write(f"**{feat.capitalize()}:** {cluster_stats.loc[i, feat]:.2f}")
                else:
                    st.info("Please select at least 2 features for clustering")
            else:
                st.warning("Not enough audio features available for clustering")

else:
    st.info("👆 Please select 'Load from Kaggle' and click the button, or upload your own CSV file!")
    
    st.markdown("---")
    st.markdown("### How to use this app:")
    st.markdown("""
    **Option 1: Load from Kaggle (Recommended)**
    1. Select "Load from Kaggle" in the sidebar
    2. Choose a dataset from the dropdown
    3. Click "Load Dataset from Kaggle"
    4. Wait for the download to complete (cached for future use)
    5. Start exploring!
    
    **Option 2: Upload CSV**
    1. Download a Spotify dataset from Kaggle manually
    2. Select "Upload CSV file" in the sidebar
    3. Upload your CSV file
    4. Start exploring!
    
    **First time setup:**
    ```bash
    pip install streamlit pandas plotly scikit-learn numpy kagglehub
    ```
    
    **Note:** You may need to authenticate with Kaggle on first use. Follow the instructions if prompted.
    """)
    
    st.markdown("---")
    st.markdown("### Popular Kaggle Datasets:")
    st.markdown("""
    - **Spotify Tracks Dataset** - 114k+ tracks with comprehensive audio features (Recommended!)
    - **Most Streamed Spotify Songs 2024** - Recent popular tracks with streaming data
    - **Spotify 1 Million Tracks** - Massive dataset for large-scale analysis
    """)

# Footer
st.markdown("---")
st.markdown("Data source: Spotify Dataset")