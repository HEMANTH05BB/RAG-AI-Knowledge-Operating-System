import unittest
from unittest.mock import patch, MagicMock
from app.retrieval.embedding_service import EmbeddingService

class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        # Reset the cached model class attribute to ensure isolated tests
        EmbeddingService._model = None
        
        self.st_patcher = patch('app.retrieval.embedding_service.SentenceTransformer')
        self.mock_st_class = self.st_patcher.start()
        
        self.mock_st = MagicMock()
        self.mock_st.get_embedding_dimension.return_value = 384
        # Mock encode to return a numpy-like array with tolist()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.1] * 384
        self.mock_st.encode.return_value = mock_array
        
        self.mock_st_class.return_value = self.mock_st

    def tearDown(self):
        self.st_patcher.stop()
        EmbeddingService._model = None

    def test_lazy_loading(self):
        # Initializing the service should NOT instantiate SentenceTransformer
        service = EmbeddingService(model_name="test-model")
        self.mock_st_class.assert_not_called()
        
        # Calling get_dimension should trigger initialization
        dim = service.get_dimension()
        self.assertEqual(dim, 384)
        self.mock_st_class.assert_called_once_with("test-model")

    def test_get_embedding(self):
        service = EmbeddingService()
        embedding = service.get_embedding("hello world")
        
        self.assertEqual(len(embedding), 384)
        self.assertEqual(embedding[0], 0.1)
        self.mock_st.encode.assert_called_once_with("hello world")

    def test_get_embeddings_batch(self):
        service = EmbeddingService()
        
        # Override mock encode for multiple texts
        mock_arr_1 = MagicMock()
        mock_arr_1.tolist.return_value = [0.1] * 384
        mock_arr_2 = MagicMock()
        mock_arr_2.tolist.return_value = [0.2] * 384
        
        self.mock_st.encode.return_value = [mock_arr_1, mock_arr_2]
        
        embeddings = service.get_embeddings(["text one", "text two"])
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0][0], 0.1)
        self.assertEqual(embeddings[1][0], 0.2)
        self.mock_st.encode.assert_called_once_with(["text one", "text two"])

if __name__ == "__main__":
    unittest.main()
