import pandas as pd
import os
from huggingface_hub import HfApi, login
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

def main():
    # Project configuration
    HF_USERNAME = "bhagat26singh"  # Update this
    PROJECT_NAME = "tourism-mlops-project"
    DATASET_REPO = f"{HF_USERNAME}/{PROJECT_NAME}"
    
    # Authenticate with Hugging Face using token from environment
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    login(token=token)
    print("✅ Authenticated with Hugging Face")
    
    # Load the tourism dataset
    tourism_data = pd.read_csv('tourism_project/data/tourism.csv')
    print(f"📊 Dataset loaded: {tourism_data.shape}")
    
    # Register dataset on unified repository
    dataset = Dataset.from_pandas(tourism_data)
    dataset.push_to_hub(DATASET_REPO, config_name="raw_data")
    print(f"✅ Dataset registered successfully on: {DATASET_REPO}")

if __name__ == "__main__":
    main()
