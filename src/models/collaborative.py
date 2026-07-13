import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.config import CF_MODEL_WEIGHTS, CF_EMBEDDING_DIM, CF_EPOCHS, CF_BATCH_SIZE, CF_LEARNING_RATE

# Global check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Falling back to NumPy-based Matrix Factorization for Collaborative Filtering.")

# --- PyTorch Implementation ---
if HAS_TORCH:
    class CFNet(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=16):
            super(CFNet, self).__init__()
            self.user_embeddings = nn.Embedding(num_users, embedding_dim)
            self.item_embeddings = nn.Embedding(num_items, embedding_dim)
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            
            # Initialize weights
            nn.init.normal_(self.user_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_embeddings.weight, std=0.01)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            
        def forward(self, user_ids, item_ids):
            user_embeds = self.user_embeddings(user_ids)
            item_embeds = self.item_embeddings(item_ids)
            
            # Dot product
            dot = (user_embeds * item_embeds).sum(dim=1, keepdim=True)
            
            # Biases
            u_bias = self.user_bias(user_ids)
            i_bias = self.item_bias(item_ids)
            
            output = dot + u_bias + i_bias
            return output.squeeze()

    class CFDataset(Dataset):
        def __init__(self, users, items, ratings):
            self.users = torch.tensor(users, dtype=torch.long)
            self.items = torch.tensor(items, dtype=torch.long)
            self.ratings = torch.tensor(ratings, dtype=torch.float32)
            
        def __len__(self):
            return len(self.ratings)
            
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx], self.ratings[idx]

# --- Main Recommender Interface ---
class CollaborativeRecommender:
    def __init__(self, embedding_dim=CF_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.global_mean = 3.0
        self.model = None
        self.numpy_user_embeddings = None
        self.numpy_item_embeddings = None
        self.numpy_user_bias = None
        self.numpy_item_bias = None
        self.is_trained = False

    def build_mappings(self, df: pd.DataFrame):
        """Creates unique integer index mappings for candidates and jobs."""
        unique_users = df["candidate_id"].unique()
        unique_items = df["job_id"].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for idx, user in enumerate(unique_users)}
        
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for idx, item in enumerate(unique_items)}
        
        self.global_mean = float(df["rating"].mean()) if len(df) > 0 else 3.0
        print(f"CF Mappings: {len(self.user_to_idx)} candidates, {len(self.item_to_idx)} jobs.")

    def train_model(self, df: pd.DataFrame):
        """Train the collaborative filtering model on interaction dataframe."""
        self.build_mappings(df)
        
        # Prepare indices
        users_idx = df["candidate_id"].map(self.user_to_idx).values
        items_idx = df["job_id"].map(self.item_to_idx).values
        ratings = df["rating"].values - self.global_mean  # Center ratings around 0
        
        if HAS_TORCH:
            num_users = len(self.user_to_idx)
            num_items = len(self.item_to_idx)
            
            # Define PyTorch Network
            self.model = CFNet(num_users, num_items, self.embedding_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=CF_LEARNING_RATE, weight_decay=1e-4)
            
            dataset = CFDataset(users_idx, items_idx, ratings)
            dataloader = DataLoader(dataset, batch_size=CF_BATCH_SIZE, shuffle=True)
            
            self.model.train()
            print(f"Training PyTorch Collaborative Filtering model for {CF_EPOCHS} epochs...")
            for epoch in range(CF_EPOCHS):
                total_loss = 0
                for u, i, r in dataloader:
                    optimizer.zero_grad()
                    preds = self.model(u, i)
                    loss = criterion(preds, r)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(r)
                
                epoch_loss = total_loss / len(dataset)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{CF_EPOCHS} - Loss: {epoch_loss:.4f}")
            
            # Save weights
            Path(CF_MODEL_WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'user_to_idx': self.user_to_idx,
                'item_to_idx': self.item_to_idx,
                'global_mean': self.global_mean
            }, CF_MODEL_WEIGHTS)
            print(f"Model saved to {CF_MODEL_WEIGHTS}")
            self.is_trained = True
            
        else:
            # NumPy SVD Alternating Least Squares (ALS) / Gradient Descent
            print("Training NumPy Matrix Factorization...")
            num_users = len(self.user_to_idx)
            num_items = len(self.item_to_idx)
            
            # Initialize matrices
            self.numpy_user_embeddings = np.random.normal(0, 0.1, (num_users, self.embedding_dim))
            self.numpy_item_embeddings = np.random.normal(0, 0.1, (num_items, self.embedding_dim))
            self.numpy_user_bias = np.zeros(num_users)
            self.numpy_item_bias = np.zeros(num_items)
            
            # Simple SGD
            lr = CF_LEARNING_RATE
            reg = 0.05
            
            for epoch in range(CF_EPOCHS):
                indices = np.arange(len(ratings))
                np.random.shuffle(indices)
                loss = 0
                for idx in indices:
                    u = users_idx[idx]
                    i = items_idx[idx]
                    r = ratings[idx]
                    
                    # Predict
                    pred = np.dot(self.numpy_user_embeddings[u], self.numpy_item_embeddings[i]) + self.numpy_user_bias[u] + self.numpy_item_bias[i]
                    err = r - pred
                    loss += err ** 2
                    
                    # Update
                    self.numpy_user_bias[u] += lr * (err - reg * self.numpy_user_bias[u])
                    self.numpy_item_bias[i] += lr * (err - reg * self.numpy_item_bias[i])
                    
                    user_emb_temp = self.numpy_user_embeddings[u].copy()
                    self.numpy_user_embeddings[u] += lr * (err * self.numpy_item_embeddings[i] - reg * self.numpy_user_embeddings[u])
                    self.numpy_item_embeddings[i] += lr * (err * user_emb_temp - reg * self.numpy_item_embeddings[i])
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{CF_EPOCHS} - Loss: {loss/len(ratings):.4f}")
            
            self.is_trained = True

    def load_model(self) -> bool:
        """Attempts to load model weights from disk."""
        if HAS_TORCH and Path(CF_MODEL_WEIGHTS).exists():
            try:
                checkpoint = torch.load(CF_MODEL_WEIGHTS)
                self.user_to_idx = checkpoint['user_to_idx']
                self.item_to_idx = checkpoint['item_to_idx']
                self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
                self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
                self.global_mean = checkpoint['global_mean']
                
                num_users = len(self.user_to_idx)
                num_items = len(self.item_to_idx)
                
                self.model = CFNet(num_users, num_items, self.embedding_dim)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.is_trained = True
                print("Loaded PyTorch Collaborative Filtering model weights.")
                return True
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                return False
        return False

    def predict_rating(self, candidate_id: str, job_id: int) -> float:
        """Predict interest rating for a candidate and job. Handles cold-starts."""
        # Check if model is loaded/trained
        if not self.is_trained:
            if not self.load_model():
                return self.global_mean
                
        # Cold start handling
        u_idx = self.user_to_idx.get(candidate_id, None)
        i_idx = self.item_to_idx.get(job_id, None)
        
        if u_idx is None or i_idx is None:
            # If cold start, return the global mean (neutral prediction)
            return self.global_mean
            
        if HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                u_tensor = torch.tensor([u_idx], dtype=torch.long)
                i_tensor = torch.tensor([i_idx], dtype=torch.long)
                pred = self.model(u_tensor, i_tensor).item()
                # Un-center rating
                return pred + self.global_mean
        else:
            # NumPy prediction
            if self.numpy_user_embeddings is not None:
                pred = np.dot(self.numpy_user_embeddings[u_idx], self.numpy_item_embeddings[i_idx]) \
                       + self.numpy_user_bias[u_idx] + self.numpy_item_bias[i_idx]
                return pred + self.global_mean
            return self.global_mean

if __name__ == "__main__":
    # Test model code
    data = pd.DataFrame([
        {"candidate_id": "C01", "job_id": 101, "rating": 5.0},
        {"candidate_id": "C01", "job_id": 102, "rating": 3.0},
        {"candidate_id": "C02", "job_id": 101, "rating": 2.0},
        {"candidate_id": "C02", "job_id": 102, "rating": 4.0}
    ])
    
    cf = CollaborativeRecommender(embedding_dim=4)
    cf.train_model(data)
    
    # Predict
    print("Prediction C01, 101 (should be near 5):", cf.predict_rating("C01", 101))
    print("Prediction C02, 101 (should be near 2):", cf.predict_rating("C02", 101))
    print("Cold Start prediction:", cf.predict_rating("C99", 999))
