from paddleocr import PaddleOCR

class OCRManager:
    def __init__(self):
        # Initializing PaddleOCR with enable_mkldnn=False 
        # This is critical for stability on Streamlit Cloud CPU instances.
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            show_log=False,
            enable_mkldnn=False  # FIX: Disables the failing CPU optimization
        )

    def get_text_with_confidence(self, img_path):
        # result = [[ [coords, (text, score)], ... ]]
        result = self.ocr.ocr(img_path, cls=True)
        
        full_results = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                full_results.append({"text": text, "confidence": confidence})
        return full_results
