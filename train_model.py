"""Root entry point for training the Rainfall Pattern ML models.

Usage:
    python train_model.py
"""

from src.train_model import train_and_save_models

if __name__ == "__main__":
    train_and_save_models(
        data_path="data/rainfall_districtwise_daily_imd.csv",
        model_output_path="models/rainfall_model.pkl"
    )
