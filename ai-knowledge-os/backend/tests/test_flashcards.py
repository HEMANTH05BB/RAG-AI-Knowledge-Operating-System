import unittest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

class TestFlashcardsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Patch VectorStoreService
        self.vs_patcher = patch('app.api.flashcards.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        # Patch LLMService
        self.llm_patcher = patch('app.api.flashcards.LLMService')
        self.mock_llm_class = self.llm_patcher.start()
        self.mock_llm = MagicMock()
        self.mock_llm_class.return_value = self.mock_llm

    def tearDown(self):
        self.vs_patcher.stop()
        self.llm_patcher.stop()

    def test_generate_flashcards_success(self):
        # 1. Mock get_document_chunks response from vector store
        self.mock_vs.get_document_chunks.return_value = [
            {
                "id": "1",
                "payload": {"text": "FastAPI is a Python web framework designed to build APIs efficiently.", "chunk_index": 0}
            },
            {
                "id": "2",
                "payload": {"text": "It handles routing, validation, and documentation out of the box.", "chunk_index": 1}
            }
        ]
        
        # 2. Mock LLM output returning valid JSON string with flashcards
        mock_flashcards = [
            {"question": "What is FastAPI?", "answer": "A modern web framework for Python APIs."},
            {"question": "What features does it handle?", "answer": "Routing, validation, and documentation."}
        ]
        self.mock_llm.generate.return_value = json.dumps(mock_flashcards)
        
        # 3. Call endpoint
        response = self.client.post(
            "/api/flashcards",
            json={"filename": "fastapi_doc.txt", "limit": 2}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        
        self.assertEqual(res_data["filename"], "fastapi_doc.txt")
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(len(res_data["flashcards"]), 2)
        self.assertEqual(res_data["flashcards"][0]["question"], "What is FastAPI?")
        self.assertEqual(res_data["flashcards"][0]["answer"], "A modern web framework for Python APIs.")
        
        # Verify vector store call
        self.mock_vs.get_document_chunks.assert_called_once_with(
            collection_name="knowledge_base",
            filename="fastapi_doc.txt"
        )
        
        # Verify LLM call was executed with model google/gemma-4-31b-it:free
        self.mock_llm_class.assert_called_once_with(model_name="google/gemma-4-31b-it:free")
        self.mock_llm.generate.assert_called_once()
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertIn("FastAPI is a Python web framework", prompt)

    def test_generate_flashcards_markdown_wrapped_success(self):
        # 1. Mock get_document_chunks
        self.mock_vs.get_document_chunks.return_value = [
            {"id": "1", "payload": {"text": "Simple test context.", "chunk_index": 0}}
        ]
        
        # 2. Mock LLM output wrapped in markdown backticks
        self.mock_llm.generate.return_value = """```json
[
  {"question": "Q1?", "answer": "A1."}
]
```"""
        
        # 3. Call endpoint
        response = self.client.post(
            "/api/flashcards",
            json={"filename": "doc.txt"}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(len(res_data["flashcards"]), 1)
        self.assertEqual(res_data["flashcards"][0]["question"], "Q1?")

    def test_generate_flashcards_document_not_found(self):
        # Mock empty chunk list returned
        self.mock_vs.get_document_chunks.return_value = []
        
        response = self.client.post(
            "/api/flashcards",
            json={"filename": "non_existent.txt"}
        )
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found or has no chunks", response.json()["detail"])
        self.mock_llm.generate.assert_not_called()

    def test_generate_flashcards_invalid_llm_json(self):
        # Mock valid chunks
        self.mock_vs.get_document_chunks.return_value = [
            {"id": "1", "payload": {"text": "Simple test context.", "chunk_index": 0}}
        ]
        # Mock LLM returning malformed/non-JSON text
        self.mock_llm.generate.return_value = "Here are your flashcards: 1. Q1: A1"
        
        response = self.client.post(
            "/api/flashcards",
            json={"filename": "doc.txt"}
        )
        
        self.assertEqual(response.status_code, 502)
        self.assertIn("Failed to parse flashcards output", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
