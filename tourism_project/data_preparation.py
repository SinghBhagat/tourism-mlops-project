import os # Ensure os is imported
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, hf_hub_download, HfApi
import joblib
import warnings
from datasets import Dataset # Still need Dataset for pushing processed data

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

    # Add debugging prints
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir()}")

    # --- Modified Data Loading ---
    # Download the raw CSV file directly from Hugging Face Hub
    raw_data_path_in_repo = "data/raw/tourism_raw.csv"
    print(f"Attempting to download raw data from {DATASET_REPO}/{raw_data_path_in_repo}")
    try:
        raw_csv_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=raw_data_path_in_repo,
            repo_type="dataset"
        )
        print(f"✅ Raw data downloaded to: {raw_csv_path}")
        df = pd.read_csv(raw_csv_path)
        print(f"📊 Raw dataset loaded into pandas DataFrame: {df.shape}")
    except Exception as e:
        print(f"❌ Error downloading or loading raw CSV from Hugging Face Hub: {e}")
        print("Please ensure:")
        print(f"- The file '{raw_data_path_in_repo}' exists in your dataset repository '{DATASET_REPO}' on Hugging Face.")
        print("- Your HF_TOKEN has read permissions for the dataset repository.")
        raise # Re-raise the exception to fail the job

    # Data preparation
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("Removed 'Unnamed: 0' column")

    # Handle categorical variables
    categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    df_processed = df.copy()
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    print("✅ Categorical features encoded")

    # Split data
    X = df_processed.drop(['ProdTaken', 'CustomerID'], axis=1)
    y = df_processed['ProdTaken']
    # Use same random_state as in notebook for consistency
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ Data split into train ({X_train.shape}) and test ({X_test.shape}) sets")


    # Save datasets locally (optional, but good practice for debugging)
    train_df_path = 'train_data.csv'
    test_df_path = 'test_data.csv'
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)
    print(f"✅ Train and test data saved locally: {train_df_path}, {test_df_path}")


    # Upload processed datasets to unified Hugging Face repository
    api = HfApi()
    print(f"Attempting to upload processed data to {DATASET_REPO}")

    try:
        # Upload train and test CSV files to organized folders
        api.upload_file(
            path_or_fileobj=train_df_path,
            path_in_repo="data/processed/train_data.csv",
            repo_id=DATASET_REPO,
            repo_type="dataset"
        )
        print(f"✅ Uploaded processed train_data.csv to {DATASET_REPO}/data/processed/")

        api.upload_file(
            path_or_fileobj=test_df_path,
            path_in_repo="data/processed/test_data.csv",
            repo_id=DATASET_REPO,
            repo_type="dataset"
        )
        print(f"✅ Uploaded processed test_data.csv to {DATASET_REPO}/data/processed/")


        # Also push as datasets format (optional, but keeps consistency)
        train_dataset_hf = Dataset.from_pandas(train_df)
        test_dataset_hf = Dataset.from_pandas(test_df)
        train_dataset_hf.push_to_hub(DATASET_REPO, config_name="train_data_processed") # Use different config name
        test_dataset_hf.push_to_hub(DATASET_REPO, config_name="test_data_processed") # Use different config name
        print(f"✅ Pushed processed data in datasets format to {DATASET_REPO}")

    except Exception as e:
        print(f"❌ Error uploading processed datasets to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN has write permissions for the dataset repository.")
        # Don't re-raise here, as data processing is done, only upload failed


    # Save label encoders locally
    encoders_path = 'label_encoders.pkl'
    joblib.dump(label_encoders, encoders_path)
    print(f"✅ Label encoders saved locally: {encoders_path}")


    # Upload label encoders to unified Hugging Face Model Hub (or Dataset Hub if preferred)
    # It makes sense to keep encoders with the model for deployment
    MODEL_REPO = f"{HF_USERNAME}/{PROJECT_NAME}-model" # Define MODEL_REPO here for upload
    api = HfApi() # Re-initialize api

    try:
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=encoders_path,
            path_in_repo="preprocessing/label_encoders.pkl",
            repo_id=MODEL_REPO,
            repo_type="model"
        )
        print(f"✅ Label encoders uploaded to Model Hub: {MODEL_REPO}/preprocessing/")
    except Exception as e:
        print(f"❌ Error uploading label encoders to Model Hub: {e}")
        print("Please ensure your HF_TOKEN has write permissions for the model repository.")
        # Don't re-raise here


    print("✅ Data preparation completed!")


if __name__ == "__main__":
    main()
