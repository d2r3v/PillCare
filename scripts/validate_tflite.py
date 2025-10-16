"""
Validate TFLite model with sanity test using interpreter.
Ensures the converted model can perform inference correctly.
"""
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def validate_tflite_model(
    tflite_path="pillcare_model.tflite",
    test_image_path=None
):
    """
    Validate TFLite model by running inference.
    
    Args:
        tflite_path: Path to the TFLite model
        test_image_path: Optional path to a test image
    
    Returns:
        bool: True if validation passes
    """
    if not os.path.exists(tflite_path):
        print(f"❌ TFLite model not found at {tflite_path}")
        return False
    
    print(f"🔍 Loading TFLite model from {tflite_path}...")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output type: {output_details[0]['dtype']}")
    
    # Create test input
    input_shape = input_details[0]['shape']
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n🖼️  Using test image: {test_image_path}")
        img = Image.open(test_image_path).convert('RGB')
        img = img.resize((input_shape[1], input_shape[2]))
        test_input = np.array(img, dtype=np.float32) / 255.0
        test_input = np.expand_dims(test_input, axis=0)
    else:
        print(f"\n🎲 Generating random test input...")
        test_input = np.random.random(input_shape).astype(np.float32)
    
    # Run inference
    print(f"🚀 Running inference...")
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Inference successful!")
    print(f"   Output shape: {output_data.shape}")
    print(f"   Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")
    print(f"   Predicted class: {np.argmax(output_data[0])}")
    print(f"   Confidence: {np.max(output_data[0]):.4f}")
    
    # Sanity checks
    checks_passed = True
    
    # Check 1: Output is probability distribution
    output_sum = np.sum(output_data[0])
    if not (0.99 <= output_sum <= 1.01):
        print(f"⚠️  Warning: Output sum is {output_sum:.4f} (expected ~1.0)")
        checks_passed = False
    else:
        print(f"✅ Output is valid probability distribution (sum={output_sum:.4f})")
    
    # Check 2: All outputs are non-negative
    if np.any(output_data < 0):
        print(f"❌ Error: Negative values in output")
        checks_passed = False
    else:
        print(f"✅ All output values are non-negative")
    
    # Check 3: Model size
    model_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"\n📊 Model size: {model_size_mb:.2f} MB")
    if model_size_mb > 50:
        print(f"⚠️  Warning: Model is quite large (>{model_size_mb:.2f}MB)")
    
    if checks_passed:
        print(f"\n✅ All validation checks passed!")
    else:
        print(f"\n⚠️  Some validation checks failed")
    
    return checks_passed

if __name__ == "__main__":
    import sys
    
    # Check if test image path provided
    test_img = None
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    
    # Try to find a test image in the project
    if not test_img:
        possible_paths = [
            "test_images",
            "pill_dataset_split/test"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        test_img = os.path.join(path, file)
                        break
                if test_img:
                    break
    
    success = validate_tflite_model(test_image_path=test_img)
    exit(0 if success else 1)
