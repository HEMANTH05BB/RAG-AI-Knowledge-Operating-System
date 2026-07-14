import unittest
import os
import shutil
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app

class TestChatAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.collection_name = "test_chat_collection"
        
        # Patch VectorStoreService, SentenceTransformer, and requests.post
        self.vs_patcher = patch('app.api.chat.VectorStoreService')
        self.mock_vs_class = self.vs_patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs
        
        self.st_patcher = patch('app.api.chat.SentenceTransformer')
        self.mock_st_class = self.st_patcher.start()
        self.mock_st = MagicMock()
        self.mock_st.get_embedding_dimension.return_value = 384
        self.mock_st.encode.return_value.tolist.return_value = [0.1] * 384
        self.mock_st_class.return_value = self.mock_st
        
        self.req_patcher = patch('app.services.llm_service.requests.post')
        self.mock_post = self.req_patcher.start()

    def tearDown(self):
        self.vs_patcher.stop()
        self.st_patcher.stop()
        self.req_patcher.stop()

    def test_chat_success_with_context(self):
        # 1. Mock vector store search results
        self.mock_vs.search_vectors.return_value = [
            {
                "id": "1",
                "score": 0.95,
                "payload": {"text": "FastAPI is a modern web framework.", "source": "fastapi_doc.txt"}
            },
            {
                "id": "2",
                "score": 0.88,
                "payload": {"text": "It is written in Python.", "source": "python_doc.txt"}
            }
        ]
        
        # 2. Mock LLM response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "FastAPI is a Python web framework used to build APIs."
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.mock_post.return_value = mock_response
        
        # 3. Call endpoint
        response = self.client.post(
            "/api/chat",
            json={
                "message": "What is FastAPI?",
                "collection_name": self.collection_name,
                "limit": 2
            }
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["answer"], "FastAPI is a Python web framework used to build APIs.")
        self.assertEqual(len(res_data["sources"]), 2)
        
        self.assertEqual(res_data["sources"][0]["source"], "fastapi_doc.txt")
        self.assertEqual(res_data["sources"][0]["text"], "FastAPI is a modern web framework.")
        self.assertEqual(res_data["sources"][0]["score"], 0.95)
        
        self.assertEqual(res_data["sources"][1]["source"], "python_doc.txt")
        
        # Verify vector store was queried
        self.mock_vs.search_vectors.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=[0.1] * 384,
            limit=2
        )
        
        # Verify LLM call was executed with RAG context
        self.mock_post.assert_called_once()
        llm_payload = self.mock_post.call_args[1]["json"]
        user_prompt = llm_payload["messages"][1]["content"]
        self.assertIn("FastAPI is a modern web framework.", user_prompt)
        self.assertIn("It is written in Python.", user_prompt)
        self.assertIn("What is FastAPI?", user_prompt)

    def test_chat_success_empty_context(self):
        # Mock empty vector store results
        self.mock_vs.search_vectors.return_value = []
        
        # Mock LLM response for empty context
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I do not know the answer based on the context."}}]
        }
        self.mock_post.return_value = mock_response
        
        response = self.client.post(
            "/api/chat",
            json={"message": "Unrelated topic"}
        )
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["answer"], "I do not know the answer based on the context.")
        self.assertEqual(len(res_data["sources"]), 0)

if __name__ == "__main__":
    unittest.main()
