import unittest
from unittest.mock import MagicMock
from app.retrieval.indexer import Indexer

class TestIndexer(unittest.TestCase):
    def setUp(self):
        # Mock VectorStoreService and EmbeddingService
        self.mock_vs = MagicMock()
        self.mock_es = MagicMock()
        
        self.mock_es.get_dimension.return_value = 384
        self.mock_es.get_embeddings.return_value = [
            [0.1] * 384,
            [0.2] * 384
        ]
        
        self.indexer = Indexer(vector_store=self.mock_vs, embedding_service=self.mock_es)
        self.collection_name = "test_indexer_collection"

    def test_index_segments_empty(self):
        res = self.indexer.index_segments([], self.collection_name)
        self.assertEqual(res["status"], "skipped")
        self.mock_vs.upsert_vectors.assert_not_called()

    def test_index_segments_success(self):
        segments = [
            {"text": "fastapi is pythonic", "metadata": {"source": "f.txt"}},
            {"text": "qdrant is fast", "metadata": {"source": "q.txt"}}
        ]
        
        self.mock_vs.upsert_vectors.return_value = True
        
        res = self.indexer.index_segments(segments, self.collection_name)
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_indexed"], 2)
        self.assertEqual(res["collection"], self.collection_name)
        
        # Verify collection creation
        self.mock_vs.create_collection.assert_called_once_with(self.collection_name, vector_size=384)
        
        # Verify batch embedding was called
        self.mock_es.get_embeddings.assert_called_once_with(["fastapi is pythonic", "qdrant is fast"])
        
        # Verify upsert_vectors was called
        self.mock_vs.upsert_vectors.assert_called_once()
        args, _ = self.mock_vs.upsert_vectors.call_args
        points = args[1]
        
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["payload"]["text"], "fastapi is pythonic")
        self.assertEqual(points[0]["payload"]["source"], "f.txt")
        self.assertEqual(points[0]["payload"]["chunk_index"], 0)
        self.assertEqual(points[0]["vector"], [0.1] * 384)
        
        self.assertEqual(points[1]["payload"]["text"], "qdrant is fast")
        self.assertEqual(points[1]["payload"]["source"], "q.txt")
        self.assertEqual(points[1]["payload"]["chunk_index"], 1)
        self.assertEqual(points[1]["vector"], [0.2] * 384)

    def test_index_segments_failure(self):
        segments = [{"text": "data", "metadata": {"source": "d.txt"}}]
        self.mock_vs.upsert_vectors.return_value = False
        self.mock_es.get_embeddings.return_value = [[0.0] * 384]
        
        res = self.indexer.index_segments(segments, self.collection_name)
        
        self.assertEqual(res["status"], "error")
        self.assertTrue(
            "db" in res["message"].lower() or 
            "database" in res["message"].lower() or 
            "vector" in res["message"].lower()
        )

if __name__ == "__main__":
    unittest.main()
