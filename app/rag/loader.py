from pypdf import PdfReader
from pathlib import Path

class PDFLoader : 


    def __init__(self,pdf_path : Path):
        if not pdf_path.exists() : 
            raise FileNotFoundError(f"PDF not foind at {pdf_path}")
        else :
            self.pdf_path =pdf_path

    def load(self) -> str: 
        reader=PdfReader(self.pdf_path) 
        pages_text=[] 

        for page in reader.pages:
            text=page.extract_text()
            if text:
                pages_text.append(text)

        return "\n".join(pages_text)        