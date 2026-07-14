from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.url import router as url_router
from app.api.search import router as search_router
from app.api.rag import router as rag_router

app = FastAPI(
    title="AI Knowledge Operating System API",
    description="Backend API for the AI Knowledge Operating System",
    version="0.1.0"
)

# Register routers
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(url_router)
app.include_router(search_router)
app.include_router(rag_router, prefix="/api", tags=["RAG"])

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from fastapi.staticfiles import StaticFiles

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

# Mount static files to serve the frontend dashboard
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
