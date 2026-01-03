import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Refined preprocessing: Reverts to Adaptive Thresholding base 
    with local contrast enhancement for partially erased text.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return image_path
        
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Local Contrast Enhancement (CLAHE) 
    # This specifically helps with "partially erased" or faded text 
    # by making the faint lines darker relative to their surroundings.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. Subtle Scaling
    # 1.5x is the "sweet spot" to increase resolution without pixelation.
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # 4. Your original Adaptive Thresholding (Optimized)
    # We use a larger block size (15) to reduce noise in the background.
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    save_path = "optimized_ocr.png"
    cv2.imwrite(save_path, processed)
    return save_path
