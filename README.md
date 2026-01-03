# Shipping Label OCR Extraction System

## Project Overview
Automated system to extract specific shipping IDs containing the `_1_` pattern from waybill images using **PaddleOCR**. [cite_start]Designed to handle degraded text with **≥75% accuracy**[cite: 10, 46].

## Technical Approach
- [cite_start]**OCR Engine**: PaddleOCR (Open-source)[cite: 43].
- [cite_start]**Preprocessing**: 2x scaling, Bilateral filtering, and Dilation to recover partially erased characters[cite: 49].
- [cite_start]**Extraction**: Regex-based fuzzy matching to handle OCR misinterpretations (e.g., mistaking `_` for `.`)[cite: 35].

## Challenges & Solutions
- **Challenge**: OCR mistaking underscores for dots on degraded labels.
- [cite_start]**Solution**: Implemented a separator-agnostic regex `[._\s-]1[._\s-]` and image dilation to thicken horizontal lines[cite: 35, 52].
- **Challenge**: Streamlit RuntimeError on CPU.
- [cite_start]**Solution**: Disabled MKLDNN optimization (`enable_mkldnn=False`) in PaddleOCR config[cite: 102].

## Performance Metrics
- [cite_start]**Target Extraction Accuracy**: >80% (tested on degraded samples)[cite: 46, 94].
- [cite_start]**Latency**: Optimized for accuracy over speed[cite: 53].

## Installation
1. `pip install -r requirements.txt`
2. Create `packages.txt` with `libgl1` and `libglib2.0-0`.
3. Run `streamlit run app.py`.
