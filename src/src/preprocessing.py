import cv2
import numpy as np

def optimize_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Rescale image: Increasing size helps OCR read smaller/degraded text
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Denoising: Removes graininess from low-quality label photos
    img = cv2.fastNlMeansDenoising(img, h=10)
    
    # 3. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 4. Adaptive Thresholding: Converts to pure black and white 
    # This helps with "faded" ink by making it solid black
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    
    # 5. Dilation: Slightly thickens characters to connect partially erased parts
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # Save optimized image for OCR engine
    processed_path = "processed_temp.png"
    cv2.imwrite(processed_path, img)
    return processed_path
