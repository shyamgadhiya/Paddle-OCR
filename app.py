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

# --- OPTIMIZED PREPROCESSING FOR DEGRADED TEXT ---
def optimize_for_ocr(image_path):
    # Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Scale up 2x to help distinguish '_' from '.'
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Adaptive thresholding to handle partially erased ink
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    # Dilation to thicken characters
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    processed_path = f"proc_{os.path.basename(image_path)}"
    cv2.imwrite(processed_path, dilated)
    return processed_path

@st.cache_resource
def load_ocr():
    return OCRManager()

ocr_tool = load_ocr()

st.title("📦 Waybill OCR Extractor")
st.write("Extracting target lines containing '_1_' pattern.")

uploaded_file = st.file_uploader("Upload Shipping Label Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    unique_filename = f"temp_{uuid.uuid4().hex}.png"
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(uploaded_file), caption="Uploaded Label", use_container_width=True)

    # OCR processing trigger
    if st.button("Process OCR"):
        with st.spinner("Analyzing document..."):
            try:
                # Preprocess for better accuracy on degraded labels
                proc_path = optimize_for_ocr(unique_filename)
                
                # Perform OCR
                results = ocr_tool.get_text_with_confidence(proc_path)
                
                # Extract target line
                target_text, confidence = extract_target_line(results)

                st.divider()
                
                # Format JSON result for deliverable
                output_data = {
                    "filename": uploaded_file.name,
                    "target_line": target_text if target_text else "Not Found",
                    "confidence": round(float(confidence), 4),
                    "success": True if target_text else False
                }

                if target_text:
                    st.subheader("Target Text Found")
                    st.success(f"**{target_text}**")
                    st.info(f"Confidence: {confidence:.2f}")
                else:
                    st.error("Target pattern '_1_' not found.")

                st.write("### Output JSON")
                st.json(output_data)

                if os.path.exists(proc_path):
                    os.remove(proc_path)

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(unique_filename):
                    os.remove(unique_filename)
