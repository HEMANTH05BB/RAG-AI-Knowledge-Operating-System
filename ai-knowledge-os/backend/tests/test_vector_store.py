import unittest
import os
import shutil
from app.services.vector_store import VectorStoreService

class TestVectorStoreService(unittest.TestCase):
    def setUp(self):
        self.db_path = "data/test_qdrant"
        self.service = VectorStoreService(db_path=self.db_path)
        self.collection_name = "test_collection"
        
    def tearDown(self):
        # Clean up database files after tests
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            
    def test_create_collection_and_upsert_search(self):
        # Create collection
        created = self.service.create_collection(self.collection_name, vector_size=3)
        self.assertTrue(created)
        
        # Re-creating should return False or be idempotent without crashing
        re_created = self.service.create_collection(self.collection_name, vector_size=3)
        self.assertFalse(re_created)
        
        # Upsert vectors
        points = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"text": "apple", "category": "fruit"}},
            {"id": 2, "vector": [0.9, 0.8, 0.7], "payload": {"text": "banana", "category": "fruit"}},
            {"id": 3, "vector": [0.1, 0.1, 0.9], "payload": {"text": "dog", "category": "animal"}}
        ]
        upserted = self.service.upsert_vectors(self.collection_name, points)
        self.assertTrue(upserted)
        
        # Search vectors
        results = self.service.search_vectors(self.collection_name, [0.1, 0.2, 0.3], limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 1) # Apple should be closest to itself
        
        # Search with filter
        filtered_results = self.service.search_vectors(
            self.collection_name, 
            [0.1, 0.2, 0.3], 
            limit=2, 
            filter_dict={"category": "animal"}
        )
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(filtered_results[0]["payload"]["text"], "dog")

if __name__ == "__main__":
    unittest.main()
