from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List
from langchain.schema import Document

class VectorStoreManager:
    def __init__(self,index,metadata):
        self.index = index
        self.metadata = metadata
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def load_index(self):
        index = faiss.read_index("data/pdf.index")
        with open("data/pdf_metadata.pkl","rb") as f:
            metadata = pickle.load(f)
        self.index = index
        self.metadata = metadata

    def search(self,query,k):
        model = self.model
        query_embedding = model.encode([query],convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx< len(self.metadata):
                results.append(self.metadata[idx])
        return results


