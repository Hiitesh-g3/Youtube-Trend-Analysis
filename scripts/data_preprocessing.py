# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

def preprocess_data():
    """
    Preprocess the raw_data.csv and save processed_data.csv
    """
    raw_data_path = "../data/raw_data.csv"
    processed_data_path = "../data/processed_data.csv"

    # Load the raw dataset
    df = pd.read_csv(raw_data_path)

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.fillna({
        'tags': 'No Tags',
        'description': 'No Description'
    }, inplace=True)

    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # Drop rows where publish_time could not be converted
    df.dropna(subset=['publish_time'], inplace=True)

    # Extract new datetime features
    df['publish_day'] = df['publish_time'].dt.day
    df['publish_month'] = df['publish_time'].dt.month
    df['publish_hour'] = df['publish_time'].dt.hour

    # Normalize numerical fields
    numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode category_id (categorical feature)
    if 'category_id' in df.columns:
        le = LabelEncoder()
        df['category_id_encoded'] = le.fit_transform(df['category_id'])

    # Save the processed data
    os.makedirs("../data", exist_ok=True)
    df.to_csv(processed_data_path, index=False)

    print("Processed data saved to ../data/processed_data.csv")

if __name__ == "__main__":
    preprocess_data()
