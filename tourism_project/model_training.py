import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from huggingface_hub import login, HfApi
import joblib
import mlflow
import mlflow.sklearn
import os

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
    
    # Load datasets from unified repository
    train_dataset = load_dataset(DATASET_REPO, name="train_data")
    test_dataset = load_dataset(DATASET_REPO, name="test_data")
    
    train_df = train_dataset['train'].to_pandas()
    test_df = test_dataset['train'].to_pandas()
    
    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"📊 Test data: {X_test.shape}")
    
    # Train model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({'accuracy': accuracy, 'f1_score': f1})
        mlflow.sklearn.log_model(model, "model")
        
        # Save model
        joblib.dump(model, 'best_model.pkl')
        
        # Upload to unified Hugging Face Model Hub with organized structure
        api = HfApi()
        
        try:
            api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj='best_model.pkl',
                path_in_repo='models/best_model.pkl',
                repo_id=MODEL_REPO,
                repo_type="model"
            )
            print(f"✅ Model uploaded to: {MODEL_REPO}")
        except Exception as e:
            print(f"❌ Error uploading model: {e}")
        
        print(f"🎯 Model trained! Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
