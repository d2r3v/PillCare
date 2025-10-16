#!/usr/bin/env python3
"""
Phase 1 Verification Script
Checks all requirements for Phase 1 completion.
"""
import os
import sys
import json
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def check_pass(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def check_fail(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def check_warn(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def check_requirements():
    """Check if requirements.txt exists and is not empty"""
    print_header("Task 1: Requirements File")
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            content = f.read().strip()
            if content:
                lines = [l for l in content.split('\n') if l.strip() and not l.startswith('#')]
                check_pass(f"requirements.txt exists with {len(lines)} packages")
                return True
            else:
                check_fail("requirements.txt is empty")
                return False
    else:
        check_fail("requirements.txt not found")
        return False

def check_folders():
    """Check if required folders exist"""
    print_header("Task 2: Folder Structure")
    
    required_folders = [
        "pill_dataset_split/",
        "pill_dataset_split/train",
        "pill_dataset_split/val", 
        "pill_dataset_split/test",
        "ocr_dataset_epillid/",
        "models/",
        "logs/",
        "plots/"
    ]
    
    all_exist = True
    for folder in required_folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            # Count items in directory
            try:
                items = len(os.listdir(folder))
                check_pass(f"{folder} exists ({items} items)")
            except:
                check_pass(f"{folder} exists")
        else:
            check_fail(f"{folder} not found")
            all_exist = False
    
    return all_exist

def check_visual_model():
    """Check if visual classification model exists and metrics saved"""
    print_header("Task 3: Visual Classification Model")
    
    checks_passed = 0
    
    # Check model file
    if os.path.exists("best_model.h5"):
        size_mb = os.path.getsize("best_model.h5") / (1024 * 1024)
        check_pass(f"best_model.h5 exists ({size_mb:.2f} MB)")
        checks_passed += 1
    else:
        check_fail("best_model.h5 not found - need to run models/train.py")
    
    # Check metrics JSON
    if os.path.exists("logs/training_metrics.json"):
        with open("logs/training_metrics.json", 'r') as f:
            metrics = json.load(f)
            check_pass(f"training_metrics.json exists")
            if 'test_accuracy' in metrics:
                check_pass(f"Test accuracy: {metrics['test_accuracy']:.4f}")
            if 'num_classes' in metrics:
                check_pass(f"Number of classes: {metrics['num_classes']}")
            checks_passed += 1
    else:
        check_fail("logs/training_metrics.json not found - need to run models/train.py")
    
    # Check training plots
    if os.path.exists("plots/training_finetuned_plots.png"):
        check_pass("Training plots saved in plots/")
        checks_passed += 1
    else:
        check_warn("Training plots not found in plots/ directory")
    
    return checks_passed >= 2

def check_tflite():
    """Check if TFLite model exists and is valid"""
    print_header("Task 4: TFLite Conversion")
    
    if os.path.exists("pillcare_model.tflite"):
        size_mb = os.path.getsize("pillcare_model.tflite") / (1024 * 1024)
        check_pass(f"pillcare_model.tflite exists ({size_mb:.2f} MB)")
        
        # Try to validate it
        check_warn("Run 'python scripts/validate_tflite.py' to test inference")
        return True
    else:
        check_fail("pillcare_model.tflite not found - run models/convert_to_tflite.py")
        return False

def check_label_map():
    """Check if label_map.json exists"""
    print_header("Task 5: Label Map")
    
    if os.path.exists("data/label_map.json"):
        with open("data/label_map.json", 'r') as f:
            label_map = json.load(f)
            check_pass(f"label_map.json exists with {len(label_map)} classes")
            
            # Print classes
            print(f"\n   Classes:")
            for idx, label in sorted({int(k): v for k, v in label_map.items()}.items()):
                print(f"      {idx}: {label}")
            return True
    else:
        check_fail("data/label_map.json not found")
        print(f"   {Colors.YELLOW}Run: python scripts/generate_label_map.py{Colors.END}")
        return False

def check_pipeline():
    """Check if pipeline script exists and is updated"""
    print_header("Task 6: Pipeline Integration")
    
    if os.path.exists("scripts/run_pipeline.py"):
        with open("scripts/run_pipeline.py", 'r') as f:
            content = f.read()
            if 'load_label_map' in content:
                check_pass("run_pipeline.py exists and uses label_map")
            else:
                check_warn("run_pipeline.py exists but may need updates")
        
        check_warn("Run 'python scripts/run_pipeline.py' to test end-to-end")
        return True
    else:
        check_fail("scripts/run_pipeline.py not found")
        return False

def main():
    """Run all Phase 1 verification checks"""
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"  🔍 PillCare Phase 1 Verification")
    print(f"{'='*60}{Colors.END}\n")
    
    results = {
        "Requirements": check_requirements(),
        "Folders": check_folders(),
        "Visual Model": check_visual_model(),
        "TFLite": check_tflite(),
        "Label Map": check_label_map(),
        "Pipeline": check_pipeline()
    }
    
    # Summary
    print_header("Phase 1 Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for task, status in results.items():
        if status:
            check_pass(f"{task}: Complete")
        else:
            check_fail(f"{task}: Incomplete")
    
    print(f"\n{Colors.BOLD}Progress: {passed}/{total} tasks completed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Phase 1 Complete! Ready for Phase 2{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}📋 Next Steps:{Colors.END}")
        
        if not results["Requirements"]:
            print(f"   1. Check requirements.txt file")
        
        if not results["Visual Model"]:
            print(f"   2. Train visual model: python models/train.py")
        
        if not results["TFLite"]:
            print(f"   3. Convert to TFLite: python models/convert_to_tflite.py")
        
        if not results["Label Map"]:
            print(f"   4. Generate label map: python scripts/generate_label_map.py")
        
        print(f"   5. Test pipeline: python scripts/run_pipeline.py")
        print()
        return 1

if __name__ == "__main__":
    exit(main())
