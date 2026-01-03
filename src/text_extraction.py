
import re

def extract_target_line(ocr_results):
    """
    Extracts the complete text line containing the pattern "_1_"[cite: 9, 34].
    Optimized for degraded labels where underscores are misread as '.', '-', or ' '.
    """
    # Pattern Logic:
    # 1. Look for a long sequence of digits (usually 10+ digits for shipping IDs).
    # 2. Match '1' flanked by any common separator: underscore, dot, hyphen, or space.
    # 3. [._\s-] treats these characters as interchangeable to handle OCR noise.
    fuzzy_pattern = r".*\d{10,}[._\s-]1[._\s-]?.*"
    
    # Secondary fallback for shorter IDs or cases where only one side has a separator
    fallback_pattern = r".*(_1_|1_|_1).*"

    for item in ocr_results:
        text = item['text'].strip()
        
        # Check fuzzy pattern first (highest accuracy for IDs)
        if re.search(fuzzy_pattern, text):
            # Normalize the output: Convert misread dots/spaces back to underscores
            # to match the expected format: "163233702292313922_1_IWV".
            normalized_text = text.replace(".", "_").replace(" ", "_").replace("-", "_")
            return normalized_text, item['confidence']
        
        # Fallback for simpler patterns
        if re.search(fallback_pattern, text):
            return text, item['confidence']
            
    return None, 0.0
