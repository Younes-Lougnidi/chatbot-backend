import os
from sys import meta_path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from multipart import file_path
from pdf2image import convert_from_path
import cv2
import numpy as np
from sqlalchemy.testing.suite.test_reflection import metadata
from test_unstructured.metrics.test_evaluate import \
    test_TextExtractionMetricsCalculator_process_document_returns_the_correct_doctype


class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
    def _process_structured_pdf(self,file_path:str)->List[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return self.text_splitter.split_documents(pages)

    def _process_unstructured_pdf(self,file_path:str)-> List[Document]:
        documents = []
        images = convert_from_path(file_path,dpi= 300)
        for i,image in enumerate(images):
            cleaned_image = self._preprocess_image(image)
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

if __name__ == "__main__":
    processsor = PDFProcessor()
    documents = processsor.process_pdf("")
    print(f"\nExtracted {len(documents)} chunks")
    for i, doc in enumerate(documents[:3]):
        print(f"\nChunk{i+1}")
        print(doc.page_content[:200]+"...")
        print(f"Source : {doc.metadata['source']},Page: {doc.metadata['page']}")