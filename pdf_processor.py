from typing import List
from langchain.schema import Document
import ftfy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np



class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )

    def _process_structured_pdf(self, file_path: str) -> List[Document]:
        try:
            import pdfplumber
            documents = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = ftfy.fix_text(text)
                        cleaned_text = cleaned_text.encode('utf-8').decode('utf-8')
                        documents.append(Document(
                            page_content=cleaned_text,
                            metadata={"source": file_path, "page": i + 1, "processing_method": "pdfplumber"}
                        ))
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            # fallback to pytesseract or other method here
            return self._process_unstructured_pdf(file_path)

    def _process_unstructured_pdf(self,file_path:str)-> List[Document]:
        documents = []
        images = convert_from_path(file_path,dpi= 300)
        for i,image in enumerate(images):
            cleaned_image = self._preprocess_image(image)
            custom_config = r'--oem 3 --psm 6 -l fra+eng'
            text = pytesseract.image_to_string(cleaned_image)
            documents.append(Document(
                page_content=text,
                metadata={
                    "source" : file_path,
                    "page" : i + 1,
                    "processing_method" : "OCR"

                }
            ))
        return self.text_splitter.split_documents(documents)

    def _preprocess_image(self,image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    def process_pdf(self,file_path)->List[Document]:
        try:
            try:
                print("Trying normal text extraction")
                return self._process_structured_pdf(file_path)
            except Exception as e :
                print(f"Falling back to OCR : {e}")
                return self._process_unstructured_pdf(file_path)
        except Exception as e :
            print(f"Failed to process {file_path}: {e}")
            return []

