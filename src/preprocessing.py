import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Optimized preprocessing to handle degraded shipping labels.
    Focuses on contrast and character edge definition.
    """
    # Load image
    img = cv2.imread(image_path)
    
    # 1. Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Rescaling (1.5x is often safer than 2x for OCR to prevent pixelation)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # 3. Contrast Enhancement (CLAHE) - Essential for partially erased text
    # This helps recover faded ink without adding global noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 4. Bilateral Filtering
    # Preserves sharp edges of characters while smoothing background noise.
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 5. Adaptive Thresholding (Reverted from Otsu)
    # Better for real-world documents with varying lighting conditions.
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    # 6. Subtle Morphological Opening (Instead of Dilation)
    # This removes small "pepper" noise dots without thickening and merging letters.
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Save for OCR engine processing
    save_path = "optimized_for_ocr.png"
    cv2.imwrite(save_path, processed)
    return save_path
