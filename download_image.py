import requests
import os
import json
from datetime import datetime

def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Write the image to file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Successfully downloaded image to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def main():
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # URL of Sheikh Mujibur Rahman image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Sheikh_Mujibur_Rahman_in_1950.jpg"
    # Note: Using the full-size image instead of thumbnail for better quality
    
    # Path to save the image
    save_path = "images/Sheikh_Mujibur_Rahman_in_1950.jpg"
    
    # Download the image
    success = download_image(image_url, save_path)
    
    if success:
        # Create image metadata file
        image_metadata = {
            "filename": os.path.basename(save_path),
            "source_url": image_url,
            "description": "Sheikh Mujibur Rahman in 1950",
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": "Wikimedia Commons",
            "attribution": "Wikimedia Commons",
            "included_in_dataset": True
        }
        
        # Save metadata
        metadata_path = "dataset/image_metadata.json"
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(image_metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Created image metadata at {metadata_path}")
        
        # Copy the image to the dataset directory for uploading to Hugging Face
        dataset_image_path = "dataset/images"
        os.makedirs(dataset_image_path, exist_ok=True)
        
        # Copy using Python instead of OS commands for better cross-platform compatibility
        import shutil
        shutil.copy2(save_path, os.path.join(dataset_image_path, os.path.basename(save_path)))
        
        print(f"Copied image to dataset directory at {dataset_image_path}")

if __name__ == "__main__":
    main()
