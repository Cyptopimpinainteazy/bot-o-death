from ml_model import train_ml_models
from data import enrich_data
from config import CHAINS
import os

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

if __name__ == "__main__":
    print("Training ML Models for Bot X3...")
    df = enrich_data(CHAINS)
