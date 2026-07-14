import unittest
import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app

class TestUploadAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_upload_collection"
        
        # Patch ProcessingPipeline in the API namespace
        self.pipeline_patcher = patch('app.api.upload.ProcessingPipeline')
        self.mock_pipeline_class = self.pipeline_patcher.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_class.return_value = self.mock_pipeline

    def tearDown(self):
        self.pipeline_patcher.stop()

    def test_upload_success_txt(self):
        # Mock process_file response
        self.mock_pipeline.process_file.return_value = {
            "status": "success",
            "filename": "test_upload.txt",
            "collection": self.collection_name,
            "total_chunks": 5,
            "summary": "This is a document summary."
        }
        
        file_content = b"This is a test content for the upload API verification."
        file_like = io.BytesIO(file_content)
        
        response = self.client.post(
            "/api/upload",
            files={"file": ("test_upload.txt", file_like, "text/plain")},
            data={
                "collection_name": self.collection_name,
                "chunk_size": 20,
                "chunk_overlap": 5,
                "generate_summary": "true"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("Successfully ingested", json_data["message"])
        self.assertEqual(json_data["filename"], "test_upload.txt")
        self.assertEqual(json_data["collection"], self.collection_name)
        self.assertEqual(json_data["chunks_count"], 5)
        self.assertEqual(json_data["summary"], "This is a document summary.")
        
        # Verify the pipeline was called with correct parameters
        self.mock_pipeline.process_file.assert_called_once()
        kwargs = self.mock_pipeline.process_file.call_args[1]
        self.assertEqual(kwargs["collection_name"], self.collection_name)
        self.assertEqual(kwargs["chunk_size"], 20)
        self.assertEqual(kwargs["chunk_overlap"], 5)
        self.assertEqual(kwargs["generate_summary"], True)

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
