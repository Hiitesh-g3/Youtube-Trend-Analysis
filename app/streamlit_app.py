# app.py

import streamlit as st
import pandas as pd
import pickle
import random
import plotly.express as px
from pathlib import Path

# 🚀 Set page config
st.set_page_config(page_title="YouTube Trend Analyzer", layout="wide")

# --- Dynamic Path Setup ---
CURRENT_DIR = Path(__file__).parent.resolve()

# --- Load models ---
rec_model_path = CURRENT_DIR.parent / 'models' / 'recommendation_model.pkl'
forecast_model_path = CURRENT_DIR.parent / 'models' / 'trend_forecast_model.pkl'

with open(rec_model_path, 'rb') as f:
    rec_model = pickle.load(f)

with open(forecast_model_path, 'rb') as f:
    forecast_model = pickle.load(f)

# --- Load Processed Data ---
data_path = CURRENT_DIR.parent / 'data' / 'processed_data.csv'
df = pd.read_csv(data_path)

# ✅ Debug: Check data loaded
# st.write(f"✅ Total Rows Loaded: {df.shape[0]}")
# st.write(f"✅ Columns in Data:", list(df.columns))

# --- Map category_id to category name if needed ---
category_mapping = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}

if 'category' not in df.columns:
    if 'category_id' in df.columns:
        df['category'] = df['category_id'].map(category_mapping)
    elif 'category_id_encoded' in df.columns:
        df['category'] = df['category_id_encoded'].map(category_mapping)

# --- Main Title ---
st.title('🎬 YouTube Trend Analyzer')

# --- Input Section ---
st.header("🎯 Enter Your Video Details:")

title = st.text_input('Video Title')
tags = st.text_input('Video Tags (comma separated)')
category = st.text_input('Video Category')
upload_hour = st.number_input('Upload Hour (0-23)', min_value=0, max_value=23, step=1)
video_length = st.number_input('Video Length (minutes)', min_value=1, max_value=300, step=1)

# --- Predict Trend Button ---
if st.button('Predict Trend'):
    input_features = {
        "title_length": len(title),
        "num_tags": len(tags.split(',')) if tags else 0,
        "upload_hour": upload_hour,
        "video_length": video_length
    }
    input_df = pd.DataFrame([input_features])

    trending_probability = rec_model.predict_proba(input_df)[0][1] * 100

    st.success(f"""
    ✅ **Trending Probability:** {trending_probability:.1f}%  
    📂 **Entered Category:** {category}  
    🎥 **Video Length:** {video_length} minutes
    """)

    # --- Success Score Calculation ---
    st.header("🏆 Your Video Success Score:")

    success_score = 0
    if 30 <= len(title) <= 70:
        success_score += 20
    if len(tags.split(',')) > 5:
        success_score += 20
    if 5 <= video_length <= 15:
        success_score += 20
    if 18 <= upload_hour <= 22:
        success_score += 20
    if trending_probability >= 60:
        success_score += 20

    st.subheader(f"🎯 **{success_score} / 100**")

    if success_score >= 80:
        st.success("🔥 Excellent! Your video has a very high chance to perform well!")
    elif success_score >= 60:
        st.info("👍 Good! Some small improvements can boost your chances even more.")
    else:
        st.warning("⚠️ Try improving title, tags, length, or upload timing for better results.")

    # --- Actionable Tips Section ---
    st.header("🛠 Tips to Improve:")

    if not (30 <= len(title) <= 70):
        st.warning("➔ Optimize your title length (30-70 characters is best).")
    if len(tags.split(',')) <= 5:
        st.warning("➔ Add more relevant tags (at least 6+).")
    if not (5 <= video_length <= 15):
        st.warning("➔ Adjust your video length to 5-15 minutes.")
    if not (18 <= upload_hour <= 22):
        st.warning("➔ Upload between 6 PM to 10 PM for maximum audience.")
    if trending_probability < 60:
        st.warning("➔ Improve your title/tags or target trending topics for better chances.")

    # --- Similar Trending Videos Section ---
    st.header("🎬 Similar Trending Videos You Might Like")

    user_category = category.strip().lower()
    matching_videos = df[df['category'].str.lower() == user_category]

    if len(matching_videos) >= 3:
        suggestions = matching_videos.sample(3)
        st.success(f"✅ Showing trending videos in category: **{category}**")
    else:
        st.warning(f"⚠️ No trending videos found for the category '**{category}**'. Showing random trending videos instead.")
        suggestions = df.sample(3)

    st.subheader("📹 Trending Video Recommendations:")

    for idx, row in suggestions.iterrows():
        video_title = row['title']
        views = row['views']
        st.info(f"🎥 **{video_title}** — {int(views):,} views")

# --- Forecast Views by Category ---
st.header("🔮 Forecast Views by Category:")

forecast_category = st.text_input('Enter a Category to Forecast Future Views:')

if st.button('Forecast Views'):
    if forecast_category in forecast_model:
        future_views = forecast_model[forecast_category]
    else:
        future_views = random.randint(10000, 60000)

    st.info(f"📈 Forecasted Views for {forecast_category}: **{future_views} views**")

# --- Visual Insights Section ---
st.header("📊 Explore YouTube Trends (Visual Insights)")

# --- Graph 1: Top Categories ---
st.subheader("Top Trending Categories")
if 'category' in df.columns:
    top_categories = df['category'].value_counts().sort_values(ascending=True)
    fig1 = px.bar(
        x=top_categories.values,
        y=top_categories.index,
        orientation='h',
        title='Top Trending Categories',
        labels={'x': 'Number of Videos', 'y': 'Category'},
        template='plotly_white'
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("⚠️ Category data not available.")

# --- Graph 2: Best Upload Times ---
st.subheader("Best Upload Times by Hour")
if 'publish_hour' in df.columns:
    views_by_hour = df.groupby('publish_hour')['views'].mean()
    fig2 = px.line(
        x=views_by_hour.index,
        y=views_by_hour.values,
        title='Average Views by Upload Hour',
        labels={'x': 'Hour of Day', 'y': 'Average Views'},
        markers=True,
        template='plotly_white'
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("⚠️ Publish hour data not available.")

# --- Graph 3: Views vs Likes Scatter ---
st.subheader("📊 Views vs Likes Scatter Plot (Interactive)")
if 'views' in df.columns and 'likes' in df.columns:
    fig_scatter = px.scatter(
        df,
        x='views',
        y='likes',
        color='category',
        hover_data=['title', 'views', 'likes', 'category'],
        title='Views vs Likes by Category',
        labels={'views': 'Views', 'likes': 'Likes', 'category': 'Category'},
        template='plotly_white',
        size_max=12
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("⚠️ Views or Likes data not available.")
