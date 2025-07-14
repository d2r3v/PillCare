import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# --- Config ---
CSV_PATH = "data/Pillbox.csv"
IMAGE_DIR = "pill_dataset_split"  # Point to the existing split dataset
OUTPUT_DIR = "ocr_dataset"
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RANDOM_STATE = 42

# --- Main ---
def create_dataset(image_dir, metadata_path, output_dir):
    """
    Creates a dataset for OCR training by matching images from the ePillID dataset
    with metadata from Pillbox.csv using NDC codes.

    Args:
        image_dir (str): Path to the directory containing ePillID images.
        metadata_path (str): Path to the Pillbox.csv metadata file.
        output_dir (str): Path to the directory where the OCR dataset will be saved.
    """
    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path, low_memory=False, dtype={'ndc9': str, 'splimprint': str})
    metadata = metadata[metadata['ndc9'].notna()]
    print(f"Loaded metadata with {len(metadata)} rows.")

    # Get all image file paths
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    except FileNotFoundError:
        print(f"Error: Image directory not found at '{image_dir}'")
        return

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    print(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("No images found. Please check the image_directory path.")
        return

    processed_count = 0
    ndcs_to_check = [os.path.basename(p).split('_')[0] for p in image_paths[:5]]
    print(f"Checking first 5 NDC codes from filenames: {ndcs_to_check}")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # Extract NDC code from filename (e.g., '0002-3228_0_0.jpg' -> '0002-3228')
        ndc_code = filename.split('_')[0].replace('-', '')

        # Find matching rows in metadata
        match = metadata[metadata['ndc9'] == ndc_code]

        if not match.empty:
            imprint_text = match.iloc[0]['splimprint']
            print(f"Match found for NDC {ndc_code}. Imprint: '{imprint_text}'")
            
            if pd.notna(imprint_text) and imprint_text.strip():
                # Copy image
                new_img_path = os.path.join(images_output_dir, filename)
                shutil.copy(img_path, new_img_path)

                # Create label file
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write(imprint_text)
                
                processed_count += 1
                if processed_count < 5:
                    print(f"  -> Processed {filename} with label '{imprint_text}'")

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {processed_count} images with valid imprint text.")
    print(f"Dataset created in: {output_dir}")
    print(f"---------------------------")

if __name__ == "__main__":
    # Update these paths to point to the new ePillID dataset
    image_directory = "data/ePillID_data/classification_data/segmented_nih_pills_224"
    metadata_file = "data/Pillbox.csv"
    output_directory = "ocr_dataset_epillid"
    
    create_dataset(image_directory, metadata_file, output_directory)
