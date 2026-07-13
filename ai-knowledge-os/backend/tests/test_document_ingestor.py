import unittest
import os
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.vector_store import VectorStoreService
from app.ingestion.document_ingestor import DocumentIngestor

class TestDocumentIngestor(unittest.TestCase):
    def setUp(self):
        # Patch SentenceTransformer to run fully offline with mock embeddings
        self.model_patcher = patch('app.ingestion.document_ingestor.SentenceTransformer')
        self.mock_transformer_class = self.model_patcher.start()
        
        self.mock_model = MagicMock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        self.mock_model.encode.side_effect = lambda text: np.array([0.1] * 384)
        self.mock_transformer_class.return_value = self.mock_model

        # Set up paths for test resources and test vector store
        self.test_dir = "data/test_ingest_resources"
        self.db_path = "data/test_ingest_qdrant"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a test text file
        self.txt_path = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("This is a test document. It is used to test the text extraction, chunking, and vector database ingestion pipeline. Let's verify everything works correctly.")
            
        # Create a test HTML file
        self.html_path = os.path.join(self.test_dir, "test.html")
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Hello World</h1><p>This is a paragraph in the html file for testing BeautifulSoup parsing.</p></body></html>")
            
        self.vector_store = VectorStoreService(db_path=self.db_path)
        self.ingestor = DocumentIngestor(vector_store=self.vector_store)
        self.collection_name = "test_ingest_collection"
        
    def tearDown(self):
        # Close database connection
        if hasattr(self, "vector_store") and self.vector_store:
            self.vector_store.client.close()
            
        # Stop model patcher
        if hasattr(self, "model_patcher"):
            self.model_patcher.stop()

        # Clean up files and directory structure after tests
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            
    def test_extract_text_txt(self):
        sections = self.ingestor.extract_text(self.txt_path)
        self.assertEqual(len(sections), 1)
        self.assertIn("This is a test document", sections[0]["text"])
        self.assertEqual(sections[0]["metadata"]["source"], "test.txt")
        
    def test_extract_text_html(self):
        sections = self.ingestor.extract_text(self.html_path)
        self.assertEqual(len(sections), 1)
        self.assertIn("Hello World", sections[0]["text"])
        self.assertIn("BeautifulSoup parsing", sections[0]["text"])
        self.assertEqual(sections[0]["metadata"]["source"], "test.html")
        
    def test_chunk_text(self):
        text = "abcdefghijklmnopqrstuvwxyz" # 26 chars
        # Chunk size 10, overlap 2
        # Chunk 1: "abcdefghij" (0:10)
        # Chunk 2: "ij" + "klmnopqrs" = "ijklmnopqr" (8:18)
        # Chunk 3: "qr" + "stuvwxyz" = "qrstuvwxyz" (16:26)
        chunks = self.ingestor.chunk_text(text, chunk_size=10, chunk_overlap=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "ijklmnopqr")
        self.assertEqual(chunks[2], "qrstuvwxyz")

    def test_ingest_file_e2e(self):
        # Ingest a text file
        result = self.ingestor.ingest_file(
            self.txt_path, 
            collection_name=self.collection_name,
            chunk_size=50, # very small chunk size to force multiple chunks
            chunk_overlap=10
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filename"], "test.txt")
        self.assertEqual(result["collection"], self.collection_name)
        self.assertTrue(result["total_chunks"] > 1)
        
        # Verify chunks exist in Vector Store
        # Let's search using the vector store
        query_vector = [0.1] * self.ingestor.vector_size
        search_res = self.vector_store.search_vectors(self.collection_name, query_vector, limit=5)
        self.assertTrue(len(search_res) > 0)
        self.assertIn("source", search_res[0]["payload"])
        self.assertEqual(search_res[0]["payload"]["source"], "test.txt")

if __name__ == "__main__":
    unittest.main()
