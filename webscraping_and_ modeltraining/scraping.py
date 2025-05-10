# Install required packages:
# pip install google_images_search pillow pandas

from google_images_search import GoogleImagesSearch
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import random

# You'll need to set up Google Custom Search API:
# 1. Create a Google Cloud Project
# 2. Enable Custom Search API
# 3. Create API credentials (API key)
# 4. Create a Custom Search Engine at https://cse.google.com/cse/all
# 5. In the CSE settings, enable "Search the entire web" and "Image search"

# Your API credentials
GCS_DEVELOPER_KEY = 'AIzaSyCstqNYXP-BxR6C5f2qqEuiZIBlGnUt3cE'
GCS_CX = '40c4fa9e2f6914d07'

# Initialize the Google Images Search client
gis = GoogleImagesSearch(GCS_DEVELOPER_KEY, GCS_CX)

# Function to create dataset directory structure
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# List of footballers to scrape (expand as needed)
footballers = [
      "Emiliano martinez","Angel di maria","Rodrigo De Paul","Antoine Griezmann","Julian Alvarez","Lautaro Martinez","Alexis Mac Allister"
]

# Base directory for the dataset
base_dir = "footballer_dataset"
create_directory(base_dir)

# DataFrame to track image metadata
metadata = []

# Configuration for image search
def search_and_download(footballer_name, num_images=30):
    print(f"Downloading images for {footballer_name}...")
    
    # Create player directory
    player_dir = os.path.join(base_dir, footballer_name.replace(" ", "_"))
    create_directory(player_dir)
    
    # Search parameters
    search_params = {
        'q': footballer_name,
        'num': num_images,
        'safe': 'medium',
        'fileType': 'jpg|png',
        'imgType': 'face',  # Focus on faces
        'imgSize': 'large'  # Prefer larger images
    }
    
    # Perform the search
    try:
        gis.search(search_params=search_params)
        
        # Download and process each image
        for i, image in enumerate(gis.results()):
            try:
                # Get image data
                raw_image = image.get_raw_data()
                
                # Process image - resize for consistency
                img = Image.open(BytesIO(raw_image))
                
                # Only keep RGB images (convert RGBA if needed)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Resize to reasonable dimensions for face recognition
                img = img.resize((256, 256), Image.LANCZOS)
                
                # Save the image
                filename = f"{footballer_name.replace(' ', '_')}_{i+1}.jpg"
                file_path = os.path.join(player_dir, filename)
                img.save(file_path, "JPEG", quality=95)
                
                # Add to metadata
                metadata.append({
                    'filename': filename,
                    'player': footballer_name,
                    'path': file_path,
                    'source_url': image.url,
                    'width': img.width,
                    'height': img.height
                })
                
                print(f"  Downloaded {filename}")
                
                # Random delay to avoid hitting API limits
                time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                print(f"  Error processing image {i+1} for {footballer_name}: {e}")
        
    except Exception as e:
        print(f"Error searching for {footballer_name}: {e}")
    
    print(f"Completed downloading images for {footballer_name}")
    # Add a short delay between players
    time.sleep(2)

# Main execution
def main():
    # Download images for each footballer
    for footballer in footballers:
        search_and_download(footballer, num_images=25)  # Adjust number as needed
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(base_dir, "dataset_metadata.csv"), index=False)
    
    print(f"Dataset creation complete. Total images: {len(metadata)}")
    print(f"Dataset saved to: {os.path.abspath(base_dir)}")
    
    # Generate a simple class mapping file
    class_mapping = {i: name for i, name in enumerate(footballers)}
    pd.DataFrame.from_dict(class_mapping, orient='index', columns=['player_name']).to_csv(
        os.path.join(base_dir, "class_mapping.csv")
    )

if __name__ == "__main__":
    main()
