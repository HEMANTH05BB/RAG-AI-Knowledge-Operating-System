import unittest
from app.retrieval.chunker import TextChunker

class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_split_by_character_empty(self):
        self.assertEqual(self.chunker.split_by_character(""), [])

    def test_split_by_character_basic(self):
        text = "abcdefghijklmnopqrstuvwxyz" # 26 chars
        # size 10, overlap 2
        chunks = self.chunker.split_by_character(text, chunk_size=10, chunk_overlap=2)
        # Expected chunks:
        # 1: abcdefghij (start 0, end 10)
        # 2: ijklmnopqr (start 8, end 18)
        # 3: qrstuvwxyz (start 16, end 26)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "ijklmnopqr")
        self.assertEqual(chunks[2], "qrstuvwxyz")

    def test_split_recursively_paragraph(self):
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
        # If chunk_size is 30, it should split at paragraphs "\n\n"
        chunks = self.chunker.split_recursively(text, chunk_size=30, chunk_overlap=5)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is paragraph one.")
        self.assertEqual(chunks[1], "This is paragraph two.")
        self.assertEqual(chunks[2], "This is paragraph three.")

    def test_split_recursively_sentences(self):
        text = "FastAPI is fast. Python is readable. Qdrant is powerful."
        # If size is 25, it should split near sentence boundaries
        chunks = self.chunker.split_recursively(text, chunk_size=25, chunk_overlap=5)
        # Expected to keep "FastAPI is fast.", "Python is readable.", "Qdrant is powerful." mostly separate
        self.assertTrue(len(chunks) >= 3)
        self.assertIn("FastAPI is fast.", chunks[0])
        self.assertIn("Python is readable.", chunks[1])
        self.assertIn("Qdrant is powerful.", chunks[-1])

    def test_split_recursively_long_words(self):
        # Word longer than chunk_size
        text = "supercalifragilisticexpialidocious" # 34 chars
        chunks = self.chunker.split_recursively(text, chunk_size=10, chunk_overlap=2)
        # Should split on fallback characters
        self.assertTrue(len(chunks) > 1)
        self.assertEqual("".join(chunks), text)

if __name__ == "__main__":
    unittest.main()
