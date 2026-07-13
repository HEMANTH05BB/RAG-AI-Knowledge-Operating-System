import unittest
import os
import shutil
import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app

class TestUploadAPI(unittest.TestCase):
    def setUp(self):
        # Set up paths for test vector store
        self.db_path = "data/test_upload_qdrant"
        self.collection_name = "test_upload_collection"
        self.client = TestClient(app)
        
        # Patch VectorStoreService and SentenceTransformer to run completely offline
        self.vs_patcher = patch('app.api.upload.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        self.st_patcher = patch('app.ingestion.document_ingestor.SentenceTransformer')
        self.mock_st_class = self.st_patcher.start()
        self.mock_st = MagicMock()
        self.mock_st.get_embedding_dimension.return_value = 384
        self.mock_st.encode.return_value.tolist.return_value = [0.1] * 384
        self.mock_st_class.return_value = self.mock_st

    def tearDown(self):
        self.vs_patcher.stop()
        self.st_patcher.stop()
        # Clean up database files if they were created
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_upload_success_txt(self):
        # Mock vector store responses
        self.mock_vs.create_collection.return_value = True
        self.mock_vs.upsert_vectors.return_value = True
        
        file_content = b"This is a test content for the upload API verification."
        file_like = io.BytesIO(file_content)
        
        response = self.client.post(
            "/api/upload",
            files={"file": ("test_upload.txt", file_like, "text/plain")},
            data={
                "collection_name": self.collection_name,
                "chunk_size": 20,
                "chunk_overlap": 5
            }
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("Successfully ingested", json_data["message"])
        self.assertEqual(json_data["filename"], "test_upload.txt")
        self.assertEqual(json_data["collection"], self.collection_name)
        self.assertTrue(json_data["chunks_count"] > 1)
        
        # Verify the ingestor successfully attempted to create the collection and upsert vectors
        self.mock_vs.create_collection.assert_called_once_with(self.collection_name, vector_size=384)
        self.mock_vs.upsert_vectors.assert_called_once()

    def test_upload_unsupported_file_type(self):
        file_content = b"some random binary content"
        file_like = io.BytesIO(file_content)
        
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.exe", file_like, "application/octet-stream")},
            data={"collection_name": self.collection_name}
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
