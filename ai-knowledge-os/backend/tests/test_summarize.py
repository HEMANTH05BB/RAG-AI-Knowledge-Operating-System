import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

class TestSummarizeAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Patch VectorStoreService
        self.vs_patcher = patch('app.api.summarize.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        # Patch LLMService
        self.llm_patcher = patch('app.api.summarize.LLMService')
        self.mock_llm_class = self.llm_patcher.start()
        self.mock_llm = MagicMock()
        self.mock_llm_class.return_value = self.mock_llm

    def tearDown(self):
        self.vs_patcher.stop()
        self.llm_patcher.stop()

    def test_summarize_success(self):
        # 1. Mock get_document_chunks response from vector store
        self.mock_vs.get_document_chunks.return_value = [
            {
                "id": "1",
                "payload": {"text": "FastAPI is a modern python web framework.", "chunk_index": 0}
            },
            {
                "id": "2",
                "payload": {"text": "It is extremely fast and high performance.", "chunk_index": 1}
            }
        ]
        
        # 2. Mock LLM generation output
        self.mock_llm.generate.return_value = "Title: FastAPI Overview\nTakeaways:\n- Fast\n- Modern\nSummary: A web framework."
        
        # 3. Call endpoint
        response = self.client.post(
            "/api/summarize",
            json={"filename": "fastapi_doc.txt"}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        
        self.assertEqual(res_data["filename"], "fastapi_doc.txt")
        self.assertEqual(res_data["status"], "success")
        self.assertIn("FastAPI Overview", res_data["summary"])
        
        # Verify vector store call
        self.mock_vs.get_document_chunks.assert_called_once_with(
            collection_name="knowledge_base",
            filename="fastapi_doc.txt"
        )
        
        # Verify LLM call was executed with the combined context
        self.mock_llm.generate.assert_called_once()
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertIn("FastAPI is a modern python web framework.", prompt)
        self.assertIn("It is extremely fast and high performance.", prompt)

    def test_summarize_document_not_found(self):
        # Mock empty chunk list returned
        self.mock_vs.get_document_chunks.return_value = []
        
        response = self.client.post(
            "/api/summarize",
            json={"filename": "non_existent.txt"}
        )
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found or has no chunks", response.json()["detail"])
        self.mock_llm.generate.assert_not_called()

if __name__ == "__main__":
    unittest.main()
