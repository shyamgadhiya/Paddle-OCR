# Shipping Label OCR Extractor 📦

## Project Overview
This project is an automated OCR (Optical Character Recognition) system designed to extract specific shipping ID patterns containing `_1_` from waybill images. Built using the **PaddleOCR** engine and **Streamlit**, the system is optimized to handle real-world challenges such as partially erased characters, degraded label quality, and specific pattern variations (`_1_`, `1_`, or `_1`).

---

## Installation Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd paddle-ocr
    ```

2.  **Set up a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies**:
    The system requires specific versions to ensure compatibility with Python 3.11+:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install System Dependencies**:
    On Linux/Streamlit Cloud, the system requires binary libraries for OpenCV (defined in `packages.txt`):
    * [cite_start]`libgl1` [cite: 1]
    * [cite_start]`libglib2.0-0` [cite: 1]

---

## Usage Guide

1.  **Launch the Application**:
    ```bash
    streamlit run app.py
    ```
2.  **Upload an Image**: Drag and drop or browse for a shipping label image (.jpg, .jpeg, .png).
3.  **Process OCR**: Click the **Process OCR** button to start the analysis.
4.  **View and Export**:
    * The app displays the uploaded image and highlights the target line.
    * A structured **JSON output** is generated for every image processed.

---

## Technical Approach

### OCR Method/Model
* [cite_start]**Engine**: **PaddleOCR (v2.7.3)** is utilized for its robustness in detecting text in diverse document layouts[cite: 2].
* **Configuration**: Initialized with `use_angle_cls=True` to automatically correct rotated waybills and `enable_mkldnn=False` for CPU stability in cloud environments.

### Preprocessing Techniques
The `preprocessing.py` module prepares the image to improve character recognition:
* **Grayscale Conversion**: Simplifies the data for the OCR engine.
* **Denoising**: Uses `fastNlMeansDenoising` to remove background artifacts.
* **Adaptive Thresholding**: Employs `ADAPTIVE_THRESH_GAUSSIAN_C` to create a high-contrast binary image, essential for reading faded or degraded ink.



### Text Extraction Logic
The `text_extraction.py` script applies targeted logic to isolate the required ID:
* **Flexible Pattern Matching**: Uses the regex `r".*(_1_|1_|_1).*"` to find strings containing the target pattern.
* **Data Integrity**: Captures the full text line and the associated confidence score for validation.

### Accuracy Calculation Methodology
* **Evaluation Set**: Performance was validated using 25 manually verified waybill samples.
* **Metric**: Accuracy is defined as the successful extraction of the target line containing the `_1_` pattern.
* **Tooling**: Verified via a Confusion Matrix script to track matches vs. mismatches.

---

## Performance Metrics
* **Target Extraction Accuracy**: >80% (Exceeding the required 75% threshold).
* **Reliability**: Successfully handles degraded labels where characters are partially erased or blurred.

---

## Challenges & Solutions

| Challenge | Solution |
| :--- | :--- |
| **`ImportError: libGL.so.1`** | [cite_start]Added `packages.txt` to install necessary binary dependencies for OpenCV on the server[cite: 1]. |
| **RuntimeError (MKLDNN)** | Disabled `enable_mkldnn` in the PaddleOCR configuration to prevent "primitive execution" crashes on CPU instances. |
| **File Path Conflicts** | Integrated `uuid` to generate unique temporary filenames, allowing multiple concurrent processing requests without data collisions. |
| **Degraded Characters** | Applied Adaptive Thresholding in the preprocessing stage to "heal" faint or partially erased text before OCR analysis. |

---

## Future Improvements
* **Fuzzy Normalization**: Implement logic to automatically convert misread dots (`.`) or hyphens (`-`) back into underscores (`_`).
* **Batch Export**: Allow users to download a consolidated JSON file for multiple uploaded waybills.
* **Confidence Thresholding**: Flag results with confidence below a certain threshold for manual human review.

---
