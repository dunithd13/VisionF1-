import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score

# Your existing model training code
def train_and_save_models():
    """Train models and save them for faster loading"""
    
    # Data folder path
    BASE_PATH = r"C:\Users\dunit\Desktop\VisionF1"
    DATA_PATH = os.path.join(BASE_PATH, "data")
    
    # Load your existing training code here
    # (Copy your existing data loading and preprocessing code)
    
    # Load datasets
    def load_dataset(filename):
        try:
            path = os.path.join(DATA_PATH, filename)
            df = pd.read_csv(path)
            print(f"Successfully loaded {filename}")
            return df
        except FileNotFoundError:
            print(f"Error: {filename} not found in {DATA_PATH}")
            return None
        except