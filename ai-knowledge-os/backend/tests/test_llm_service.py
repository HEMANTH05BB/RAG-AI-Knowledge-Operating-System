import unittest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService

class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.openrouter_key = "sk-or-v1-testkey123456"
        self.google_key = "AIzaSyTestKey123"
        self.openrouter_model = "google/gemini-2.5-flash"
        self.google_model = "gemini-2.5-flash"

    def test_provider_detection_openrouter(self):
        service = LLMService(api_key=self.openrouter_key, model_name=self.openrouter_model)
        self.assertEqual(service.provider, "openrouter")
        self.assertEqual(service.base_url, "https://openrouter.ai/api/v1/chat/completions")

    def test_provider_detection_google(self):
        service = LLMService(api_key=self.google_key, model_name=self.google_model)
        self.assertEqual(service.provider, "google")
        self.assertIn("generativelanguage.googleapis.com", service.base_url)

    @patch('app.services.llm_service.requests.post')
    def test_call_openrouter_payload_and_parsing(self, mock_post):
        # Mock successful OpenRouter response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a response from OpenRouter."
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        service = LLMService(api_key=self.openrouter_key, model_name=self.openrouter_model)
        res = service.generate_response(prompt="Hello", system_instruction="You are a poet", temperature=0.5)
        
        self.assertEqual(res, "This is a response from OpenRouter.")
        
        # Verify post payload
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.openrouter_key}")
        
        payload = kwargs["json"]
        self.assertEqual(payload["model"], self.openrouter_model)
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "You are a poet"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hello"})

    @patch('app.services.llm_service.requests.post')
    def test_call_google_payload_and_parsing(self, mock_post):
        # Mock successful Google Gemini response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "This is a response from Google Gemini."
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        service = LLMService(api_key=self.google_key, model_name=self.google_model)
        res = service.generate_response(prompt="Hi", system_instruction="You are a coding assistant", temperature=0.2)
        
        self.assertEqual(res, "This is a response from Google Gemini.")
        
        # Verify post payload
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("generativelanguage.googleapis.com", args[0])
        self.assertIn(f"key={self.google_key}", args[0])
        
        payload = kwargs["json"]
        self.assertEqual(payload["generationConfig"]["temperature"], 0.2)
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Hi")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "You are a coding assistant")

    @patch('app.services.llm_service.requests.post')
    def test_rag_generation_structure(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "RAG Answer"}}]
        }
        mock_post.return_value = mock_response
        
        service = LLMService(api_key=self.openrouter_key, model_name=self.openrouter_model)
        res = service.generate_rag_response(
            query="What is Qdrant?",
            contexts=["Qdrant is a vector database.", "It stores embeddings."]
        )
        
        self.assertEqual(res, "RAG Answer")
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        
        # Check system prompt and context inclusion
        self.assertIn("RAG Answer", res)
        self.assertIn("system", payload["messages"][0]["role"])
        user_prompt = payload["messages"][1]["content"]
        self.assertIn("Qdrant is a vector database.", user_prompt)
        self.assertIn("It stores embeddings.", user_prompt)
        self.assertIn("What is Qdrant?", user_prompt)

if __name__ == "__main__":
    unittest.main()
