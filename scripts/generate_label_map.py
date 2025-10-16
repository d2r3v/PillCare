"""
Generate label_map.json from training dataset classes.
This ensures consistency between training and inference.
"""
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_label_map(dataset_dir="pill_dataset_split", output_path="data/label_map.json"):
    """
    Generate label_map.json from training dataset structure.
    
    Args:
        dataset_dir: Path to the split dataset directory
        output_path: Where to save the label_map.json file
    """
    train_dir = os.path.join(dataset_dir, "train")
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Training directory not found at {train_dir}")
        return None
    
    # Use ImageDataGenerator to get class labels in correct order
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # Extract class mapping
    class_indices = generator.class_indices
    # Invert to get index -> label mapping
    label_map = {v: k for k, v in class_indices.items()}
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"✅ Label map saved to {output_path}")
    print(f"   Total classes: {len(label_map)}")
    print(f"   Classes: {list(label_map.values())}")
    
    return label_map

if __name__ == "__main__":
    label_map = generate_label_map()
    if label_map:
        print(f"\n📋 Label Map Preview:")
        for idx, label in sorted(label_map.items()):
            print(f"   {idx}: {label}")
