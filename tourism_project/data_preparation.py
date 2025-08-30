import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, load_dataset
from huggingface_hub import login
import joblib
import os

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
    
    # Load dataset from unified Hugging Face repository
    #tourism_project/data/tourism.csv
    #dataset = load_dataset("your-username/tourism-mlops-project", data_files="data/raw/tourism_raw.csv")
    dataset = load_dataset(DATASET_REPO, name="raw_data")
    df = dataset['train'].to_pandas()
    print(f"📊 Dataset loaded from: {DATASET_REPO}")
    
    # Data preparation
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Handle categorical variables
    categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    df_processed = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Split data
    X = df_processed.drop(['ProdTaken', 'CustomerID'], axis=1)
    y = df_processed['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Upload to unified Hugging Face repository with organized structure
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset.push_to_hub(DATASET_REPO, config_name="train_data")
    test_dataset.push_to_hub(DATASET_REPO, config_name="test_data")
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print(f"✅ Data preparation completed and uploaded to: {DATASET_REPO}")

if __name__ == "__main__":
    main()
