# scripts/data_collection.py

import pandas as pd
import glob
import os

def download_and_save_data():
    """
    Load multiple Kaggle YouTube trending datasets from /data/USvideos/
    and save the combined data as raw_data.csv
    """
    # Dynamically find the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    usvideos_dir = os.path.join(project_root, "data", "USvideos")
    output_path = os.path.join(project_root, "data", "raw_data.csv")

    # Find all CSV files inside /data/USvideos/
    csv_files = glob.glob(os.path.join(usvideos_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found inside /data/USvideos/")
        return

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            df_list.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined raw data saved to {output_path}")
    else:
        print("No valid CSV files to combine.")

if __name__ == "__main__":
    download_and_save_data()
