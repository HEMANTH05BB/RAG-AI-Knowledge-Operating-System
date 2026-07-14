import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

class TestRAGAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Patch the global instances directly in the rag module namespace
        self.mock_indexer = MagicMock()
        self.indexer_patcher = patch('app.api.rag.indexer', self.mock_indexer)
        self.indexer_patcher.start()
        
        self.mock_es = MagicMock()
        self.mock_es.embed.return_value = [0.03] * 384
        self.es_patcher = patch('app.api.rag.embedder', self.mock_es)
        self.es_patcher.start()
        
        self.mock_llm = MagicMock()
        self.llm_patcher = patch('app.api.rag.llm', self.mock_llm)
        self.llm_patcher.start()

    def tearDown(self):
        self.indexer_patcher.stop()
        self.es_patcher.stop()
        self.llm_patcher.stop()

    def test_ask_success(self):
        # 1. Mock search results returned by QdrantIndexer.search
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.payload = {
            "text": "Qdrant is a vector similarity search engine.",
            "source": "qdrant_wiki.txt"
        }
        self.mock_indexer.search.return_value = [mock_result]
        
        # 2. Mock LLM generation output
        self.mock_llm.generate.return_value = "Qdrant is a vector similarity search engine."
        
        # 3. Call RAG ask endpoint
        response = self.client.post(
            "/api/ask",
            json={"question": "What is Qdrant?"}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        
        self.assertEqual(res_data["question"], "What is Qdrant?")
        self.assertEqual(res_data["answer"], "Qdrant is a vector similarity search engine.")
        self.assertEqual(len(res_data["citations"]), 1)
        self.assertEqual(res_data["citations"][0]["source"], "qdrant_wiki.txt")
        self.assertEqual(res_data["citations"][0]["score"], 0.85)
        
        # Verify call arguments
        self.mock_es.embed.assert_called_once_with("What is Qdrant?")
        self.mock_indexer.search.assert_called_once_with([0.03] * 384, limit=5)
        self.mock_llm.generate.assert_called_once()
        
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertIn("Qdrant is a vector similarity search engine.", prompt)
        self.assertIn("What is Qdrant?", prompt)

    def test_ask_empty_results(self):
        self.mock_indexer.search.return_value = []
        
        response = self.client.post(
            "/api/ask",
            json={"question": "Unrelated question"}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["answer"], "No relevant information found in the knowledge base.")
        self.assertEqual(len(res_data["citations"]), 0)
        self.mock_llm.generate.assert_not_called()

if __name__ == "__main__":
    unittest.main()
