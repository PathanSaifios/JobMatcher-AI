import pandas as pd
from src.config import INTERACTIONS_CSV
from src.models.collaborative import CollaborativeRecommender

def main():
    print("Starting training of Collaborative Filtering model...")
    try:
        interactions_df = pd.read_csv(INTERACTIONS_CSV)
    except FileNotFoundError:
        print(f"Interactions file not found at {INTERACTIONS_CSV}. Please run the simulation script first.")
        return
        
    cf_recommender = CollaborativeRecommender()
    cf_recommender.train_model(interactions_df)
    print("Collaborative Filtering model training completed successfully.")

if __name__ == "__main__":
    main()
