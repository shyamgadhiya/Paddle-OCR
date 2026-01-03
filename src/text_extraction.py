import re

def extract_target_line(ocr_results):
    """
    Extracts the complete line containing pattern _1_[cite: 34].
    Handles degraded text variations: _1_, 1_, or _1.
    """
    # Regex logic: 
    # Match strings containing '1' with an underscore on either or both sides.
    # Pattern allows for alphanumeric characters surrounding the match.
    pattern = r".*(_1_|1_|_1).*"
    
    for item in ocr_results:
        text = item['text'].strip()
        if re.search(pattern, text):
            # Returns the complete text line [cite: 9]
            return text, item['confidence']
            
    return None, 0.0
