import os
from huggingface_hub import HfApi, login
import joblib
import json

def deploy_to_huggingface_spaces():
    """Deploy the Streamlit app to an existing Hugging Face Space (created manually)"""
    # Change directory to access files correctly
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Changed directory to: {os.getcwd()}")

    # Load the best model name and metadata
    try:
        with open('../model_building/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        best_model_name = model_metadata.get('model_name', 'best_model')
        # Adjust the filename based on the best model name
        model_filename = f"../model_building/best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
        print(f"Using best model file: {model_filename}")
    except Exception as e:
        print(f"Error loading model metadata: {e}")
        # Fallback to a default if metadata not found
        model_filename = "../model_building/best_model_bagging.pkl" # Assuming Bagging was best based on previous run
        print(f"Using fallback model file: {model_filename}")


    # Login to Hugging Face (make sure you have HF_TOKEN set)
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HF_TOKEN environment variable not set.")
        print("Please set it in your environment before running this script.")
        os.chdir(original_dir) # Change back to original directory
        return
    else:
        print("✅ HF_TOKEN environment variable found.")

    try:
        login(token=token)
        print("✅ Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"❌ Error logging in: {e}")
        os.chdir(original_dir) # Change back to original directory
        return

    # Initialize HF API
    api = HfApi()

    # Configuration - Use configured app repository
    # Assuming APP_REPO is defined in the Colab environment and accessible
    # If not, you might need to define it here or pass it as an argument
    # For now, let's assume it's accessible
    try:
        # Access APP_REPO from the outer scope
        repo_id = os.environ.get("APP_REPO")
        if not repo_id:
             # Fallback if not found in environment variables
             # Make sure to replace with your actual username
             repo_id = "bhagat26singh/tourism-mlops-project-app"
             print(f"⚠️ APP_REPO environment variable not found, using default: {repo_id}")
        print(f"Deploying to Hugging Face Space: {repo_id}")

    except NameError:
        # If APP_REPO is not defined in the outer scope (e.g., running script directly)
        # Define it here or pass as an argument
        repo_id = "bhagat26singh/tourism-mlops-project-app" # Make sure to replace with your actual username
        print(f"⚠️ APP_REPO not defined in outer scope, using default: {repo_id}")


    try:
        # Upload files to organized folders in the Space
        files_to_upload = [
            ("app.py", "app.py"),
            ("requirements.txt", "requirements.txt"),
            (model_filename, "models/best_model.pkl"), # Use the dynamically determined model file
            ("../model_building/label_encoders.pkl", "preprocessing/label_encoders.pkl"), # Add label encoders
            ("../model_building/model_metadata.json", "metadata/model_metadata.json"), # Add metadata
            ("Dockerfile", "Dockerfile"), # Keep Dockerfile if needed for Space build
            ("README.md", "README.md"), # Add README if it exists
        ]
        for src, dest in files_to_upload:
            if os.path.exists(src):
                api.upload_file(
                    path_or_fileobj=src,
                    path_in_repo=dest,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"✅ Uploaded {src} to {dest} in Space {repo_id}")
            else:
                print(f"❌ File not found: {src}")

        print(f"\n🌐 Your app should be available at: https://huggingface.co/spaces/{repo_id}")
        print("⏳ Please allow a few minutes for the Space to build and deploy.")
        print("👀 Check the build logs on the Hugging Face Space page for status.")

    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        print("Please check the following:")
        print(f"- The Space '{repo_id}' exists on Hugging Face and is set to 'Streamlit' SDK.")
        print("- Your HF_TOKEN has write permissions for this Space.")
        print("- File paths in the script are correct relative to where the script is run.")
    finally:
        os.chdir(original_dir) # Change back to original directory

if __name__ == "__main__":
    # Set APP_REPO as an environment variable for the script to access it
    # This is a workaround for %%writefile not having access to Colab variables directly
    os.environ["APP_REPO"] = "bhagat26singh/tourism-mlops-project-app" # Make sure to replace with your actual username
    deploy_to_huggingface_spaces()
