import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score 
from datasets import load_dataset 
from huggingface_hub import login, HfApi
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost 
import json 
import warnings 
import time 
import os
warnings.filterwarnings('ignore')

def main():
    # Project configuration
    HF_USERNAME = "bhagat26singh"  # Update this
    PROJECT_NAME = "tourism-mlops-project"
    DATASET_REPO = f"{HF_USERNAME}/{PROJECT_NAME}"
    MODEL_REPO = f"{HF_USERNAME}/{PROJECT_NAME}-model"

    # Authenticate with Hugging Face using token from environment
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    login(token=token)
    print("✅ Authenticated with Hugging Face")

    # Load datasets from processed CSV files
    # --- Modified Data Loading ---
    train_df_path = 'tourism_project/data/processed/train_data.csv'
    test_df_path = 'tourism_project/data/processed/test_data.csv'
    print(f"Attempting to load processed data from: {train_df_path}, {test_df_path}")
    try:
        # Ensure index is not loaded as a column
        train_df = pd.read_csv(train_df_path, index_col=False)
        test_df = pd.read_csv(test_df_path, index_col=False)
        print("✅ Processed data loaded from CSVs")
    except Exception as e:
        print(f"❌ Error loading processed data from CSVs: {e}")
        print("Please ensure the 'data/processed/train_data.csv' and 'data/processed/test_data.csv' files exist in your GitHub repository.")
        raise # Re-raise the exception to fail the job


    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']

    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Test data shape: {X_test.shape}")
    print(f"Training features: {X_train.columns.tolist()}") # Print feature names to verify
    print(f"Testing features: {X_test.columns.tolist()}") # Print feature names to verify


    # Model training (using XGBoost as in notebook outputs)
    # Ensure MLflow tracking is enabled
    # MLflow tracking URI should be set in the GitHub Actions workflow or environment
    # mlflow.set_tracking_uri("http://localhost:5000") # Remove or comment out if set externally

    with mlflow.start_run(run_name="XGBoost Model Training"): # Give run a name
        print("Starting MLflow run...")

        # Define and train model
        # Use the parameters from your best model in the notebook (XGBoost)
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss', # Use eval_metric from notebook outputs
            use_label_encoder=False, # Recommended for XGBoost with recent versions
            n_estimators=100, # Example parameter, use best from notebook
            learning_rate=0.1, # Example parameter, use best from notebook
            random_state=42 # Use random_state for reproducibility
            # Add other best parameters from notebook if available
        )
        print(f"Training model: {type(model).__name__}")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"✅ Model training completed in {training_time:.2f} seconds")

        # Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred) # Calculate AUC

        print(f"🎯 Model evaluated! Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")


        # Log metrics and parameters
        print("Logging metrics and parameters to MLflow...")
        mlflow.log_params({
            'model_type': type(model).__name__,
            'n_estimators': 100, # Log parameters used
            'learning_rate': 0.1,
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            # Log other parameters
        })
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_score': auc,
            'training_time': training_time,
        })
        print("✅ Metrics and parameters logged.")


        # Log model
        print("Logging model artifact...")
        # Use mlflow.xgboost.log_model for XGBoost models
        mlflow.xgboost.log_model(model, "model") # Use "model" as artifact_path (name)

        print("✅ Model artifact logged.")


        # Save model and encoders locally for potential upload outside MLflow
        print("Saving model and encoders locally...")
        model_filename = f"best_model_{type(model).__name__.lower()}.pkl"
        joblib.dump(model, model_filename)
        # Need to load label_encoders from the data_preparation step's output
        try:
            label_encoders = joblib.load('label_encoders.pkl') # Assuming this file is available from data_preparation
            joblib.dump(label_encoders, 'label_encoders.pkl') # Save it again locally
            print("✅ Label encoders saved locally: label_encoders.pkl")
        except FileNotFoundError:
            print("❌ Warning: label_encoders.pkl not found. Skipping local save/upload of encoders.")
            label_encoders = None # Set to None if not found

        print(f"✅ Model saved locally: {model_filename}")

        # Save model metadata
        model_metadata = {
            'model_name': type(model).__name__,
            'model_type': type(model).__name__, # Redundant, but good practice
            'parameters': model.get_params(), # Log all parameters
            'performance_metrics': {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auc_score': auc,
            },
            'feature_names': X_train.columns.tolist(), # Log feature names used
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open('model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=4)
        print("✅ Model metadata saved locally.")


    # Upload best model and encoders to unified Hugging Face Model Hub
    # This part is separate from MLflow artifact logging but needed for deployment
    api = HfApi()

    try:
        print(f"Attempting to upload model and encoders to {MODEL_REPO}")
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

        # Upload the locally saved model file
        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=f'models/{model_filename}', # Save in models folder
            repo_id=MODEL_REPO,
            repo_type="model"
        )
        print(f"✅ Model file uploaded to: {MODEL_REPO}/models/{model_filename}")

        # Upload the locally saved label encoders file if it exists
        if label_encoders is not None:
            api.upload_file(
                path_or_fileobj='label_encoders.pkl',
                path_in_repo='preprocessing/label_encoders.pkl', # Save in preprocessing folder
                repo_id=MODEL_REPO,
                repo_type="model"
            )
            print(f"✅ Label encoders uploaded to: {MODEL_REPO}/preprocessing/label_encoders.pkl")
        else:
             print("Skipping label encoder upload as file was not found locally.")

        # Upload model metadata
        api.upload_file(
            path_or_fileobj='model_metadata.json',
            path_in_repo='metadata/model_metadata.json', # Save in metadata folder
            repo_id=MODEL_REPO,
            repo_type="model"
        )
        print(f"✅ Model metadata uploaded to: {MODEL_REPO}/metadata/model_metadata.json")


    except Exception as e:
        print(f"❌ Error uploading model artifacts to Hugging Face Model Hub: {e}")
        print("Please ensure your HF_TOKEN has write permissions for the model repository.")
        # Don't re-raise here, as model is saved locally and logged in MLflow


    print("✅ Model training completed!")

if __name__ == "__main__":
    main()
