import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

class TestRAGAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_rag_collection"
        
        # Patch dependencies
        self.vs_patcher = patch('app.api.rag.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        self.es_patcher = patch('app.api.rag.EmbeddingService')
        self.mock_es_class = self.es_patcher.start()
        self.mock_es = MagicMock()
        self.mock_es.get_embedding.return_value = [0.03] * 384
        self.mock_es_class.return_value = self.mock_es
        
        self.llm_patcher = patch('app.api.rag.LLMService')
        self.mock_llm_class = self.llm_patcher.start()
        self.mock_llm = MagicMock()
        self.mock_llm_class.return_value = self.mock_llm

    def tearDown(self):
        self.vs_patcher.stop()
        self.es_patcher.stop()
        self.llm_patcher.stop()

    def test_rag_generation_success(self):
        # 1. Mock vector store search output
        self.mock_vs.search_vectors.return_value = [
            {
                "id": "pt-1",
                "score": 0.85,
                "payload": {
                    "text": "Qdrant is a vector similarity search engine.",
                    "source": "qdrant_wiki.txt",
                    "source_type": "text_file"
                }
            },
            {
                "id": "pt-2",
                "score": 0.50, # Will be filtered out if min_score = 0.6
                "payload": {
                    "text": "It can be deployed via Docker.",
                    "source": "docker_setup.txt",
                    "source_type": "text_file"
                }
            }
        ]
        
        # 2. Mock LLM output
        self.mock_llm.generate_response.return_value = "Qdrant is a vector similarity search engine."
        
        # 3. Call endpoint with score threshold filter
        response = self.client.post(
            "/api/rag",
            json={
                "query": "What is Qdrant?",
                "collection_name": self.collection_name,
                "min_score": 0.6,
                "temperature": 0.1,
                "system_instruction": "Answer in one sentence."
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        
        # Assertions
        self.assertEqual(res_data["answer"], "Qdrant is a vector similarity search engine.")
        # Only 1 source should remain (pt-2 has score 0.50 < 0.6)
        self.assertEqual(len(res_data["sources"]), 1)
        self.assertEqual(res_data["sources"][0]["id"], "pt-1")
        self.assertEqual(res_data["sources"][0]["score"], 0.85)
        self.assertEqual(res_data["sources"][0]["source"], "qdrant_wiki.txt")
        self.assertEqual(res_data["sources"][0]["metadata"]["source_type"], "text_file")
        
        # Verify LLM call parameter formats
        self.mock_llm.generate_response.assert_called_once()
        kwargs = self.mock_llm.generate_response.call_args[1]
        self.assertEqual(kwargs["system_instruction"], "Answer in one sentence.")
        self.assertEqual(kwargs["temperature"], 0.1)
        
        prompt = kwargs["prompt"]
        self.assertIn("Qdrant is a vector similarity search engine.", prompt)
        self.assertNotIn("It can be deployed via Docker.", prompt) # Filtered out!

if __name__ == "__main__":
    unittest.main()
