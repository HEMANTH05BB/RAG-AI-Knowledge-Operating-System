import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.url import extract_youtube_video_id

class TestURLIngestionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_url_collection"
        
        # Start patches for heavy dependencies
        self.vs_patcher = patch('app.api.url.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        self.st_patcher = patch('app.api.url.SentenceTransformer')
        self.mock_st_class = self.st_patcher.start()
        self.mock_st = MagicMock()
        self.mock_st.get_embedding_dimension.return_value = 384
        self.mock_st.encode.return_value.tolist.return_value = [0.05] * 384
        self.mock_st_class.return_value = self.mock_st

    def tearDown(self):
        self.vs_patcher.stop()
        self.st_patcher.stop()

    def test_youtube_video_id_extractor(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        for url in valid_urls:
            self.assertEqual(extract_youtube_video_id(url), "dQw4w9WgXcQ")
            
        self.assertIsNone(extract_youtube_video_id("https://google.com"))

    @patch('app.api.url.fetch_youtube_transcript')
    def test_youtube_ingest_success(self, mock_fetch_yt):
        mock_fetch_yt.return_value = "Never gonna give you up, never gonna let you down."
        self.mock_vs.upsert_vectors.return_value = True
        
        response = self.client.post(
            "/api/url",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "collection_name": self.collection_name,
                "chunk_size": 20,
                "chunk_overlap": 5
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["source_type"], "youtube_transcript")
        # Chunks:
        # "Never gonna give you" (len 20)
        # " give you up, never"
        # etc.
        self.assertTrue(res_data["total_chunks"] > 0)
        
        # Verify vector store upsert was called
        self.mock_vs.upsert_vectors.assert_called_once()
        self.mock_vs.create_collection.assert_called_once_with(self.collection_name, vector_size=384)

    @patch('app.api.url.requests.get')
    def test_webpage_ingest_success(self, mock_get):
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.content = b"<html><head><title>Test Page</title></head><body><h1>Welcome to FastAPI</h1><p>FastAPI is high performance.</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        self.mock_vs.upsert_vectors.return_value = True
        
        response = self.client.post(
            "/api/url",
            json={
                "url": "https://fastapi.tiangolo.com",
                "collection_name": self.collection_name
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["source_type"], "webpage")
        self.assertTrue(res_data["total_chunks"] > 0)
        
        self.mock_vs.upsert_vectors.assert_called_once()
        mock_get.assert_called_once()

    def test_ingest_failure_invalid_url(self):
        # Trigger validation error / failure
        response = self.client.post(
            "/api/url",
            json={
                "url": "not-a-valid-url",
                "collection_name": self.collection_name
            }
        )
        # Should raise 400 bad request from fetch helper
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
