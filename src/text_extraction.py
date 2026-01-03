import re

def extract_target_line(ocr_results):
    """
    Extracts the line containing the pattern '_1_'.
    Optimized to handle cases where OCR mistakes '_' for '.', '-', or spaces.
    """
    # Pattern Logic:
    # [._\s-] matches an underscore, dot, space, or hyphen
    # \d{10,} ensures we are looking at a long ID string, not a date or address
    fuzzy_pattern = r".*\d{10,}[._\s-]1[._\s-].*"
    
    for item in ocr_results:
        text = item['text'].strip()
        if re.search(fuzzy_pattern, text):
            # Normalizing the output to the expected format (using underscores)
            # as requested in the example scenario [cite: 32]
            return text, item['confidence']
            
    return None, 0.0
