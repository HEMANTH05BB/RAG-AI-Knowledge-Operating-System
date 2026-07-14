from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.indexer import QdrantIndexer
from app.services.llm_service import LLMService

router = APIRouter()

embedder = EmbeddingService()
indexer = QdrantIndexer()
llm = LLMService()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(req: AskRequest):
    try:
        # Convert question to embedding
        query_embedding = embedder.embed(req.question)

        # Search relevant chunks from Qdrant
        results = indexer.search(query_embedding, limit=5)

        if not results:
            return {
                "question": req.question,
                "answer": "No relevant information found in the knowledge base.",
                "citations": []
            }

        context_parts = []
        citations = []

        for i, r in enumerate(results, start=1):
            text = r.payload.get("text", "")
            source = r.payload.get("source", "Unknown")

            context_parts.append(f"[{i}] Source: {source}\n{text}")

            citations.append({
                "source": source,
                "score": round(r.score, 3)
            })

        context = "\n\n".join(context_parts)

        # Build RAG prompt
        prompt = f"""
You are an AI Knowledge Operating System assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, say you do not know.

Context:
{context}

Question:
{req.question}

Provide a concise and accurate answer. Mention source numbers like [1], [2] when relevant.
"""

        # Generate answer using OpenRouter
        answer = llm.generate(prompt)

        return {
            "question": req.question,
            "answer": answer,
            "citations": citations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")
