import streamlit as st
import os
import json
import uuid  # For generating unique file names
from src.ocr_engine import OCRManager
from src.text_extraction import extract_target_line
from PIL import Image

st.set_page_config(page_title="Shipping Label OCR", page_icon="📦")

@st.cache_resource
def load_ocr():
    return OCRManager()

ocr_tool = load_ocr()

st.title("📦 Waybill OCR Extractor")

uploaded_file = st.file_uploader("Upload Shipping Label Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Use uuid to ensure every image has a unique path to avoid state errors
    unique_filename = f"temp_{uuid.uuid4().hex}.png"
    
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Label", use_container_width=True)

    if st.button("Process OCR"):
        with st.spinner("Analyzing document..."):
            try:
                # 1. Perform OCR [cite: 9]
                results = ocr_tool.get_text_with_confidence(unique_filename)
                
                # 2. Extract specific line containing pattern [cite: 10, 34]
                target_text, confidence = extract_target_line(results)

                st.divider()
                
                # Mandatory JSON output for every image [cite: 39]
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

            except Exception as e:
                st.error(f"Error during OCR: {str(e)}")
            finally:
                # Always cleanup the unique file to save disk space [cite: 51]
                if os.path.exists(unique_filename):
                    os.remove(unique_filename)
