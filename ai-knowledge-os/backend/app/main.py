from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.url import router as url_router
from app.api.search import router as search_router

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to the AI Knowledge Operating System API",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }
