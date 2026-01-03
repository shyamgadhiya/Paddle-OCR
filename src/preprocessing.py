import cv2
import numpy as np

def preprocess_for_accuracy(image_path):
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale 2x to make small dots/underscores more distinct
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Bilateral filter removes noise but preserves character edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Otsu's Binarization for optimal black/white contrast
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological dilation to "heal" partially erased underscores
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    
    save_path = "optimized_for_ocr.png"
    cv2.imwrite(save_path, processed)
    return save_path
