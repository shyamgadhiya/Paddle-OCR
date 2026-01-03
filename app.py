import streamlit as st
import os
import json
import uuid
from src.ocr_engine import OCRManager
from src.text_extraction import extract_target_line
from src.preprocessing import preprocess_image
from PIL import Image

st.set_page_config(page_title="Shipping Label OCR", page_icon="📦")

@st.cache_resource
def load_ocr():
    return OCRManager()

ocr_tool = load_ocr()

st.title("📦 Waybill OCR Extractor")
st.write("Extract specific ID patterns containing '_1_' from shipping labels.")

uploaded_file = st.file_uploader("Upload Shipping Label Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    unique_filename = f"temp_{uuid.uuid4().hex}.png"
    
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Label", use_container_width=True)

    if st.button("Process OCR"):
        with st.spinner("Analyzing document..."):
            try:
                # Optimized Preprocessing [cite: 49]
                processed_path = preprocess_image(unique_filename)
                
                # 1. Perform OCR [cite: 9]
                results = ocr_tool.get_text_with_confidence(processed_path)
                
                # 2. Extract target line [cite: 10, 34]
                target_text, target_conf = extract_target_line(results)

                st.divider()
                
                # --- HIGHLIGHTED TEXT DISPLAY  ---
                st.subheader("Extracted Document Content")
                st.write("Target line is highlighted in blue below:")
                
                for res in results:
                    current_line = res['text']
                    # If this line matches our extracted target, highlight it
                    if target_text and current_line == target_text:
                        st.info(f"👉 **{current_line}** (Target)")
                    else:
                        st.text(current_line)

                # --- JSON OUTPUT [cite: 39] ---
                output_data = {
                    "filename": uploaded_file.name,
                    "target_line": target_text if target_text else "Not Found",
                    "confidence": round(float(target_conf), 4) if target_text else 0.0,
                    "success": True if target_text else False
                }

                st.write("### Output JSON")
                st.json(output_data)

                # Cleanup processed image
                if os.path.exists(processed_path):
                    os.remove(processed_path)

            except Exception as e:
                st.error(f"Error during OCR: {str(e)}")
            finally:
                if os.path.exists(unique_filename):
                    os.remove(unique_filename)
