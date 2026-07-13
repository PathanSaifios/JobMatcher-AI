import re
import pandas as pd
from pathlib import Path
from src.config import RAW_JOBS_CSV, PROCESSED_JOBS_CSV

def clean_text(text: str) -> str:
    """Basic NLP text cleaning: lowercase, remove URLs, special characters, and extra spaces."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Replace special characters and punctuation with spaces (retaining essential programming characters like C++, C#, .NET)
    # We want to keep alphanumeric and some characters. Let's do a basic clean.
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_jobs():
    """Reads raw jobs CSV, cleans text fields, handles missing values, and saves processed CSV."""
    print("Preprocessing jobs dataset...")
    if not Path(RAW_JOBS_CSV).exists():
        raise FileNotFoundError(f"Raw jobs CSV not found at {RAW_JOBS_CSV}. Please run the data generator/download script first.")
        
    df = pd.read_csv(RAW_JOBS_CSV)
    
    # Handle missing values
    df['title'] = df['title'].fillna('Unknown Title')
    df['company'] = df['company'].fillna('Unknown Company')
    df['location'] = df['location'].fillna('Remote')
    df['description'] = df['description'].fillna('')
    df['salary'] = df['salary'].fillna('Not Disclosed')
    df['experience_level'] = df['experience_level'].fillna('Mid')
    
    # Create a clean description column
    df['clean_description'] = df['description'].apply(clean_text)
    
    # Save the processed data
    Path(PROCESSED_JOBS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_JOBS_CSV, index=False)
    print(f"Preprocessed {len(df)} jobs. Saved to {PROCESSED_JOBS_CSV}")

if __name__ == "__main__":
    preprocess_jobs()
