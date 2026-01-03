import re

def extract_target_line(ocr_results):
    """
    Optimized extraction logic to handle OCR misidentifications.
    Treats underscores, dots, and hyphens as the same character.
    """
    # Pattern: Look for long digits followed by a separator and the number 1.
    # [._\s-] matches: underscore, dot, space, or hyphen.
    pattern = r".*\d{10,}[._\s-]1[._\s-].*"
    
    for item in ocr_results:
        text = item['text'].strip()
        if re.search(pattern, text):
            # Normalization: Return the text as found, 
            # or replace separators with underscores for consistency.
            normalized = text.replace(".", "_").replace("-", "_").replace(" ", "_")
            return normalized, item['confidence']
            
    return None, 0.0
