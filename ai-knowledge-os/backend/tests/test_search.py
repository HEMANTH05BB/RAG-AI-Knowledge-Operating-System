import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

class TestSearchAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_search_collection"
        
        # Patch VectorStoreService and EmbeddingService
        self.vs_patcher = patch('app.api.search.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        self.es_patcher = patch('app.api.search.EmbeddingService')
        self.mock_es_class = self.es_patcher.start()
        self.mock_es = MagicMock()
        self.mock_es.get_embedding.return_value = [0.02] * 384
        self.mock_es_class.return_value = self.mock_es

    def tearDown(self):
        self.vs_patcher.stop()
        self.es_patcher.stop()

    def test_search_success_no_filter(self):
        # 1. Mock database output
        self.mock_vs.search_vectors.return_value = [
            {
                "id": "point-1",
                "score": 0.98,
                "payload": {
                    "text": "Deep Learning is a subset of machine learning.",
                    "source": "ml_intro.txt",
                    "chunk_index": 0,
                    "length": 46
                }
            }
        ]
        
        # 2. Call endpoint
        response = self.client.post(
            "/api/search",
            json={
                "query": "Deep Learning",
                "collection_name": self.collection_name,
                "limit": 1
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        
        self.assertEqual(res_data["query"], "Deep Learning")
        self.assertEqual(res_data["collection"], self.collection_name)
        self.assertEqual(len(res_data["results"]), 1)
        
        item = res_data["results"][0]
        self.assertEqual(item["id"], "point-1")
        self.assertEqual(item["score"], 0.98)
        self.assertEqual(item["text"], "Deep Learning is a subset of machine learning.")
        self.assertEqual(item["source"], "ml_intro.txt")
        self.assertEqual(item["metadata"]["chunk_index"], 0)
        self.assertEqual(item["metadata"]["length"], 46)
        
        # Verify calls
        self.mock_es.get_embedding.assert_called_once_with("Deep Learning")
        self.mock_vs.search_vectors.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=[0.02] * 384,
            limit=1,
            search_filter=None
        )

    def test_search_success_with_metadata_filter(self):
        self.mock_vs.search_vectors.return_value = []
        filter_dict = {"source_type": "webpage"}
        
        response = self.client.post(
            "/api/search",
            json={
                "query": "FastAPI",
                "collection_name": self.collection_name,
                "limit": 3,
                "filter_metadata": filter_dict
            }
        )
        
        self.assertEqual(response.status_code, 200)
        
        self.mock_vs.search_vectors.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=[0.02] * 384,
            limit=3,
            search_filter=filter_dict
        )

if __name__ == "__main__":
    unittest.main()
