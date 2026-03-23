import os
import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

class MedicalRetriever:
    def __init__(self, db_path="./persisted/chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        # Using a standard medical-friendly embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="medical_qa", 
            embedding_function=self.embedding_fn
        )

    def build_index(self, jsonl_path):
        """Indexes the JSONL file into ChromaDB if the collection is empty."""
        if self.collection.count() > 0:
            print(f"Index already exists with {self.collection.count()} entries.")
            return

        print(f"Indexing data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        documents = []
        metadatas = []
        ids = []

        for i, line in enumerate(lines):
            data = json.loads(line)
            # The 'text' field contains "Question: ... \nAnswer: ... \nSource: ..."
            documents.append(data['text'])
            metadatas.append({"source": data.get('source', 'Unknown')})
            ids.append(f"id_{i}")

            # Batch add to avoid memory spikes
            if len(documents) >= 500:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                documents, metadatas, ids = [], [], []
                print(f"Indexed {i+1}/{len(lines)} samples...")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        print("Indexing complete.")

    def retrieve(self, query, n_results=3):
        """Retrieves the most relevant context for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Combine documents into a single context string
        context = ""
        for doc in results['documents'][0]:
            context += f"{doc}\n\n"
        
        return context.strip()
