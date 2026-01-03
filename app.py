import streamlit as st
import os
import json
import uuid
import cv2
import numpy as np
from src.ocr_engine import OCRManager
from src.text_extraction import extract_target_line
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Shipping Label OCR", page_icon="📦")

# --- IMAGE PREPROCESSING FOR DEGRADED TEXT ---
def optimize_image_for_ocr(image_path):
    """
    Enhances contrast and thickens characters to prevent 
    underscores (_) from being misread as dots (.)[cite: 49, 50].
    """
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale 2x to help OCR with small/faded characters
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Denoise and apply Adaptive Thresholding to make characters pop 
    thresh = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Dilation to 'heal' partially erased underscores or lines
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    optimized_path = f"optimized_{os.path.basename(image_path)}"
    cv2.imwrite(optimized_path, dilated)
    return optimized_path

# --- MODEL LOADING ---
@st.cache_resource
def load_ocr():
    return OCRManager()

ocr_tool = load_ocr()

# --- UI LAYOUT ---
st.title("📦 Waybill OCR Extractor")
st.write("Optimized for degraded labels and misread underscores.")

uploaded_file = st.file_uploader("Upload Shipping Label Image", type=['jpg', 'jpeg', 'png']) [cite: 58]

if uploaded_file is not None:
    unique_filename = f"temp_{uuid.uuid4().hex}.png"
    
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Label", use_container_width=True)

    if st.button("Process OCR"): [cite: 59]
        with st.spinner("Optimizing and Analyzing document..."):
            try:
                # 1. Preprocess the image to fix character degradation [cite: 49, 53]
                processed_img_path = optimize_image_for_ocr(unique_filename)
                
                # 2. Perform OCR on the optimized image [cite: 9]
                results = ocr_tool.get_text_with_confidence(processed_img_path)
                
                # 3. Extract target using pattern logic (_1_, 1_, _1) [cite: 10, 13]
                target_text, confidence = extract_target_line(results)

                st.divider()
                
                # 4. Mandatory JSON Output [cite: 39]
                output_data = {
                    "filename": uploaded_file.name,
                    "target_line": target_text if target_text else "Not Found",
                    "confidence": round(float(confidence), 4),
                    "success": True if target_text else False,
                    "processing_applied": "2x_upscale_dilation_threshold"
                }

                if target_text:
                    st.subheader("Target Text Found") [cite: 60]
                    st.success(f"**{target_text}**")
                    st.info(f"Confidence: {confidence:.2f}")
                else:
                    st.error("Target pattern not found. Image may be too degraded.")

                st.write("### Output JSON")
                st.json(output_data)

                # Cleanup processed file
                if os.path.exists(processed_img_path):
                    os.remove(processed_img_path)

            except Exception as e:
                st.error(f"Error during OCR: {str(e)}") [cite: 52]
            finally:
                if os.path.exists(unique_filename):
                    os.remove(unique_filename) [cite: 51]
