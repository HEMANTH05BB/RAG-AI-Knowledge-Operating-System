import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.api.url import extract_youtube_video_id

class TestURLIngestionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_url_collection"
        
        # Patch ProcessingPipeline in the URL API namespace
        self.pipeline_patcher = patch('app.services.processing_pipeline.ProcessingPipeline')
        self.mock_pipeline_class = self.pipeline_patcher.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_class.return_value = self.mock_pipeline

    def tearDown(self):
        self.pipeline_patcher.stop()

    def test_youtube_video_id_extractor(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ"
        ]
        for url in valid_urls:
            self.assertEqual(extract_youtube_video_id(url), "dQw4w9WgXcQ")
            
        invalid_urls = [
            "https://example.com",
            "https://youtube.com",
            "https://youtu.be/short"
        ]
        for url in invalid_urls:
            self.assertIsNone(extract_youtube_video_id(url))

    def test_webpage_ingest_success(self):
        # Mock pipeline output for webpage
        self.mock_pipeline.process_url.return_value = {
            "status": "success",
            "url": "https://example.com/test",
            "collection": self.collection_name,
            "total_chunks": 4,
            "source_type": "webpage",
            "summary": "This is a webpage summary."
        }
        
        response = self.client.post(
            "/api/url",
            json={
                "url": "https://example.com/test",
                "collection_name": self.collection_name,
                "chunk_size": 200,
                "chunk_overlap": 20,
                "generate_summary": True
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["url"], "https://example.com/test")
        self.assertEqual(res_data["total_chunks"], 4)
        self.assertEqual(res_data["source_type"], "webpage")
        self.assertEqual(res_data["summary"], "This is a webpage summary.")
        
        # Verify pipeline call parameters
        self.mock_pipeline.process_url.assert_called_once_with(
            url="https://example.com/test",
            collection_name=self.collection_name,
            chunk_size=200,
            chunk_overlap=20,
            generate_summary=True
        )

    def test_youtube_ingest_success(self):
        self.mock_pipeline.process_url.return_value = {
            "status": "success",
            "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "collection": self.collection_name,
            "total_chunks": 12,
            "source_type": "youtube_transcript",
            "summary": "This is a youtube video summary."
        }
        
        response = self.client.post(
            "/api/url",
            json={
                "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                "collection_name": self.collection_name
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["total_chunks"], 12)
        self.assertEqual(res_data["source_type"], "youtube_transcript")
        
        # Verify call properties
        self.mock_pipeline.process_url.assert_called_once()
        kwargs = self.mock_pipeline.process_url.call_args[1]
        self.assertEqual(kwargs["url"], "https://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_ingest_failure_invalid_url(self):
        self.mock_pipeline.process_url.return_value = {
            "status": "error",
            "message": "Failed to download webpage"
        }
        
        response = self.client.post(
            "/api/url",
            json={
                "url": "invalid-url-format",
                "collection_name": self.collection_name
            }
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Failed to download webpage", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
