from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List
from langchain.schema import Document
from pdf_processor import PDFProcessor
import os

processor = PDFProcessor()
class VectorStoreManager:
    def __init__(self,index = None,metadata= None):
        self.index = index
        self.metadata = metadata
        self.model =  SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    def build_index(self,documents : List[Document]):
        model = self.model
        texts = [doc.page_content for doc in documents]
        embeddings = model.encode(texts,convert_to_numpy=True).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index,"data/pdf.index")
        metadata = [doc.metadata | {"page_content": doc.page_content} for doc in documents]
        with open("data/pdf_metadata.pkl","wb") as f:
            pickle.dump(metadata,f)
        self.index = index
        self.metadata = metadata

    def _normalize_query(self, query: str) -> str:
        import unicodedata, ftfy
        query = unicodedata.normalize("NFC", query)
        return ftfy.fix_text(query).strip()

    def load_index(self):
        index = faiss.read_index("data/pdf.index")
        with open("data/pdf_metadata.pkl","rb") as f:
            metadata = pickle.load(f)
        self.index = index
        self.metadata = metadata

    def search(self,query,k):
        model = self.model
        query = self._normalize_query(query)
        query_embedding = model.encode([query],convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx< len(self.metadata):
                results.append(self.metadata[idx])
        return results
    @staticmethod
    def load_and_process_pdfs(pdf_folder_path: str) -> List[Document]:
        all_documents = []
        for filename in os.listdir(pdf_folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(pdf_folder_path, filename)
                try:
                    docs = processor.process_pdf(file_path)
                    all_documents.extend(docs)
                    print(f"Processed {filename}:extracted {len(docs)} chunks")
                except Exception as e:
                    print(f"Failed to process {filename} : {e}")
        print(f"Total chunks extracted from all PDFs : {len(all_documents)}")
        return all_documents

if __name__ == '__main__':
    vector_store1 = VectorStoreManager(None,None)
    all_documents = vector_store1.load_and_process_pdfs("data/pdfs")
    vector_store1.build_index(all_documents)
