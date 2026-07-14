import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from app.config import settings

class VectorStoreService:
    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the Qdrant client in serverless/local-disk mode.
        Data is persisted in the specified path.
        """
        path = db_path or settings.QDRANT_PATH
        os.makedirs(path, exist_ok=True)
        self.client = QdrantClient(path=path)
        
    def create_collection(self, collection_name: str, vector_size: int, distance=rest_models.Distance.COSINE) -> bool:
        """
        Create a collection if it doesn't already exist.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                return True
            return False
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
            
    def upsert_vectors(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """
        Upsert a list of vectors/points to Qdrant.
        Each point dict should look like:
        {
            "id": int or str,
            "vector": list of floats,
            "payload": dict of metadata
        }
        """
        try:
            qdrant_points = []
            for pt in points:
                qdrant_points.append(
                    rest_models.PointStruct(
                        id=pt["id"],
                        vector=pt["vector"],
                        payload=pt.get("payload", {})
                    )
                )
            
            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
            
    def search_vectors(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest vectors.
        """
        try:
            # Build simple filter if provided
            search_filter = None
            if filter_dict:
                conditions = []
                for key, val in filter_dict.items():
                    conditions.append(
                        rest_models.FieldCondition(
                            key=key,
                            match=rest_models.MatchValue(value=val)
                        )
                    )
                search_filter = rest_models.Filter(must=conditions)
                
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=search_filter
            ).points
            
            return [
                {
                    "id": res.id,
                    "score": res.score,
                    "payload": res.payload
                }
                for res in results
            ]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
            
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
