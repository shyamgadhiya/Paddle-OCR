import os
import json
import sys
import pandas as pd
from datetime import datetime

# Add root directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_engine import OCRManager
from src.text_extraction import extract_target_line
from src.preprocessing import preprocess_image

def run_accuracy_test(data_dir="data/test_samples", gt_file="tests/ground_truth.json"):
    # Initialize components
    ocr_tool = OCRManager()
    
    # Load Ground Truth
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    results_list = []
    correct_count = 0
    
    # Ensure results directory exists
    os.makedirs("results/extracted_json_outputs", exist_ok=True)
    
    print(f"--- Starting OCR Accuracy Test: {len(ground_truth)} Samples ---")
    
    for filename, expected_text in ground_truth.items():
        img_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: {filename} not found in {data_dir}")
            continue
            
        # 1. Preprocess
        proc_path = preprocess_image(img_path)
        
        # 2. OCR and Extract
        raw_results = ocr_tool.get_text_with_confidence(proc_path)
        predicted_text, confidence = extract_target_line(raw_results)
        
        # 3. Cleanup temp file
        if os.path.exists(proc_path):
            os.remove(proc_path)
            
        # 4. Compare
        is_correct = (predicted_text == expected_text)
        if is_correct:
            correct_count += 1
            
        # 5. Create specific JSON output for this image (Requirement)
        image_result = {
            "filename": filename,
            "expected": expected_text,
            "predicted": predicted_text if predicted_text else "NOT_FOUND",
            "confidence": round(float(confidence), 4),
            "match": is_correct
        }
        
        # Save individual JSON result
        json_out_path = f"results/extracted_json_outputs/{filename.split('.')[0]}.json"
        with open(json_out_path, 'w') as jf:
            json.dump(image_result, jf, indent=4)
            
        results_list.append(image_result)
        status = "✅" if is_correct else "❌"
        print(f"{status} {filename} | Conf: {confidence:.2f}")

    # 6. Final Metrics
    total = len(results_list)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    report = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "correct_extractions": correct_count,
        "accuracy_percentage": round(accuracy, 2),
        "threshold_met": accuracy >= 75.0
    }
    
    # Save Final Report
    with open("results/accuracy_report.json", "w") as rf:
        json.dump(report, rf, indent=4)
        
    print(f"\n--- Test Complete ---")
    print(f"Accuracy: {accuracy:.2f}% | Target: 75.00%")
    print(f"Full report saved to results/accuracy_report.json")

if __name__ == "__main__":
    run_accuracy_test()
