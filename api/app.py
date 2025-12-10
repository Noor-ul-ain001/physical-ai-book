# api/app.py - Main application entry point

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
import os
import sys

# Import configuration
from .config import config

# Define the application
app = FastAPI(
    title="Physical AI & Humanoid Robotics - Backend API",
    description="AI-powered Q&A system for the interactive textbook",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers import chat_router, rag_router, auth_router, status_router
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(rag_router, prefix="/api", tags=["rag"])
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(status_router, prefix="/api", tags=["status"])

# Add static files (if needed for frontend serving)
if os.path.exists("../build"):  # Docusaurus build directory
    app.mount("/", StaticFiles(directory="../build", html=True), name="static")

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    logger.info("Starting up Physical AI & Humanoid Robotics API...")
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.warning("Configuration validation warnings:")
        for error in validation_errors:
            logger.warning(f"  - {error}")
    
    # Initialize Gemini client if API key is available
    if config.GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("Gemini client initialized")
    else:
        logger.warning("Gemini API key not found - using mock responses")
    
    # Initialize Qdrant client if available
    if config.QDRANT_URL and config.QDRANT_API_KEY:
        from qdrant_client import QdrantClient
        global qdrant_client
        qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        logger.info("Qdrant client initialized")
    else:
        logger.warning("Qdrant configuration not found - using mock RAG")
    
    logger.info("API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup services on shutdown
    """
    logger.info("Shutting down API...")

@app.get("/")
async def root():
    """
    Root endpoint for API health check
    """
    return {
        "message": "Physical AI & Humanoid Robotics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "config_valid": len(config.validate()) == 0
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "gemini_api": config.GEMINI_API_KEY is not None,
            "qdrant_db": config.QDRANT_URL is not None and config.QDRANT_API_KEY is not None,
            "database": config.DATABASE_URL is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app", 
        host=config.HOST, 
        port=config.PORT, 
        reload=config.DEBUG
    )