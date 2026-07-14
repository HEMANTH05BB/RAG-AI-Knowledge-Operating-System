import unittest
from unittest.mock import MagicMock, patch
from app.services.processing_pipeline import ProcessingPipeline

class TestProcessingPipeline(unittest.TestCase):
    def setUp(self):
        # Create mock dependencies
        self.mock_indexer = MagicMock()
        self.mock_chunker = MagicMock()
        self.mock_llm = MagicMock()
        
        # Configure embedding model dim in indexer
        self.mock_indexer.embedding_service.model_name = "test-model"
        
        self.pipeline = ProcessingPipeline(
            indexer=self.mock_indexer,
            chunker=self.mock_chunker,
            llm=self.mock_llm
        )

    @patch('app.services.processing_pipeline.DocumentIngestor')
    def test_process_file_no_summary(self, mock_ingestor_class):
        # Mock document ingestor instance
        mock_ingestor = MagicMock()
        mock_ingestor_class.return_value = mock_ingestor
        
        # Override mock ingestor extract_text
        mock_ingestor.extract_text.return_value = [
            {"text": "page 1 contents", "metadata": {"page_number": 1, "source": "test.txt"}}
        ]
        
        # Configure pipeline's internal document ingestor to use our mock
        self.pipeline.document_ingestor = mock_ingestor
        
        self.mock_chunker.split_recursively.return_value = ["page 1 chunk 1", "page 1 chunk 2"]
        self.mock_indexer.index_segments.return_value = {"status": "success", "total_indexed": 2}
        
        res = self.pipeline.process_file(
            file_path="/path/to/test.txt",
            collection_name="test_col",
            chunk_size=100,
            chunk_overlap=10,
            generate_summary=False
        )
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_chunks"], 2)
        self.assertIsNone(res["summary"])
        
        # Verify text chunking parameters
        self.mock_chunker.split_recursively.assert_called_once_with(
            "page 1 contents", chunk_size=100, chunk_overlap=10
        )
        
        # Verify indexer input
        self.mock_indexer.index_segments.assert_called_once()
        args, _ = self.mock_indexer.index_segments.call_args
        segments = args[0]
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "page 1 chunk 1")
        self.assertEqual(segments[0]["metadata"]["page_number"], 1)

    @patch('app.services.processing_pipeline.DocumentIngestor')
    def test_process_file_with_summary(self, mock_ingestor_class):
        mock_ingestor = MagicMock()
        mock_ingestor_class.return_value = mock_ingestor
        mock_ingestor.extract_text.return_value = [
            {"text": "document body content", "metadata": {"source": "doc.pdf"}}
        ]
        self.pipeline.document_ingestor = mock_ingestor
        
        self.mock_llm.generate_response.return_value = "This is a summary of the doc."
        self.mock_chunker.split_recursively.return_value = ["chunk 1"]
        self.mock_indexer.index_segments.return_value = {"status": "success", "total_indexed": 1}
        
        res = self.pipeline.process_file(
            file_path="/path/to/doc.pdf",
            collection_name="test_col",
            generate_summary=True
        )
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["summary"], "This is a summary of the doc.")
        
        # Verify summary LLM prompt
        self.mock_llm.generate_response.assert_called_once()
        args, kwargs = self.mock_llm.generate_response.call_args
        self.assertIn("document body content", args[0])
        self.assertEqual(kwargs["temperature"], 0.3)
        
        # Verify summary is attached to chunk metadata
        indexer_args = self.mock_indexer.index_segments.call_args[0][0]
        self.assertEqual(indexer_args[0]["metadata"]["document_summary"], "This is a summary of the doc.")

    @patch('app.services.processing_pipeline.fetch_webpage_content')
    @patch('app.services.processing_pipeline.extract_youtube_video_id')
    def test_process_url_webpage_no_summary(self, mock_yt_extractor, mock_fetch_web):
        mock_yt_extractor.return_value = None
        mock_fetch_web.return_value = "webpage raw content body"
        
        self.mock_chunker.split_recursively.return_value = ["chunk a", "chunk b"]
        self.mock_indexer.index_segments.return_value = {"status": "success", "total_indexed": 2}
        
        res = self.pipeline.process_url(
            url="https://example.com/test",
            collection_name="test_col",
            generate_summary=False
        )
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["source_type"], "webpage")
        
        self.mock_chunker.split_recursively.assert_called_once_with(
            "webpage raw content body", chunk_size=500, chunk_overlap=100
        )
        
        # Verify metadata properties passed to indexer
        indexer_args = self.mock_indexer.index_segments.call_args[0][0]
        self.assertEqual(indexer_args[0]["metadata"]["source_type"], "webpage")
        self.assertEqual(indexer_args[0]["metadata"]["url"], "https://example.com/test")

if __name__ == "__main__":
    unittest.main()
