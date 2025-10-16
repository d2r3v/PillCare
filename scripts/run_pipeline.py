"""
Full pipeline for pill identification combining visual classification and OCR.
This is the main entry point for the PillCare system.
"""
import cv2
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.classify import classify_pill

# Try to import OCR, but make it optional for now
try:
    from scripts.ocr_crnn import predict_text
    OCR_AVAILABLE = True
except Exception as e:
    print(f"⚠️  OCR not available: {e}")
    OCR_AVAILABLE = False

def load_label_map(label_map_path="data/label_map.json"):
    """Load class labels from label_map.json"""
    try:
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        # Convert string keys to integers
        return {int(k): v for k, v in label_map.items()}
    except FileNotFoundError:
        print(f"⚠️  Warning: label_map.json not found at {label_map_path}")
        print(f"   Run: python scripts/generate_label_map.py")
        return None

def run_pipeline(image_path, label_map=None, show_image=False):
    """
    Run full pill identification pipeline.
    
    Args:
        image_path: Path to pill image
        label_map: Dictionary mapping class indices to labels
        show_image: Whether to display the image
    
    Returns:
        dict: Results containing classification and OCR results
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📸 Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image")
        return None

    results = {}
    
    # --- VISUAL CLASSIFICATION ---
    try:
        print(f"\n🔍 Running visual classification...")
        class_idx, confidence = classify_pill(image)
        
        if label_map and class_idx in label_map:
            predicted_class = label_map[class_idx]
        else:
            predicted_class = f"Class_{class_idx}"
        
        results['classification'] = {
            'class_index': int(class_idx),
            'class_name': predicted_class,
            'confidence': float(confidence)
        }
        
        print(f"   ✅ Predicted: {predicted_class}")
        print(f"   📊 Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"   ❌ Classification failed: {e}")
        results['classification'] = {'error': str(e)}
    
    # --- OCR (if available) ---
    if OCR_AVAILABLE:
        try:
            print(f"\n📝 Running OCR...")
            imprint_text = predict_text(image)
            results['ocr'] = {
                'text': imprint_text,
                'length': len(imprint_text)
            }
            print(f"   ✅ Imprint: '{imprint_text}'")
        except Exception as e:
            print(f"   ⚠️  OCR failed: {e}")
            results['ocr'] = {'error': str(e)}
    else:
        results['ocr'] = {'status': 'not_available'}
        print(f"\n📝 OCR: Not available (CRNN model not trained yet)")
    
    # --- FUSION LOGIC ---
    print(f"\n🔗 Combined Results:")
    if 'classification' in results and 'class_name' in results['classification']:
        print(f"   💊 Pill: {results['classification']['class_name']}")
        print(f"   📊 Confidence: {results['classification']['confidence']:.2%}")
    
    if 'ocr' in results and 'text' in results['ocr']:
        print(f"   🔤 Imprint: {results['ocr']['text']}")
    
    return results

def main():
    """Main function to run pipeline on test images"""
    # Load label map
    label_map = load_label_map()
    
    # Find test images
    test_dirs = ["test_images", "pill_dataset_split/test", "data/test_images"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images.append(os.path.join(test_dir, file))
            if test_images:
                break
    
    if not test_images:
        print("❌ No test images found. Please add images to test_images/ directory")
        return
    
    print(f"\n🎯 Found {len(test_images)} test images")
    print(f"🚀 Starting pipeline...\n")
    
    # Process all images
    all_results = []
    for img_path in test_images[:5]:  # Limit to 5 images for demo
        result = run_pipeline(img_path, label_map=label_map)
        if result:
            all_results.append({
                'image': os.path.basename(img_path),
                'results': result
            })
    
    # Save results
    os.makedirs("logs", exist_ok=True)
    output_path = "logs/pipeline_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline completed!")
    print(f"📄 Results saved to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
