# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from datetime import datetime

# Import routers
from .routers.chat_router import chat_router
from .routers.rag_router import rag_router
from .routers.auth_router import auth_router
from .routers.status_router import status_router

# Configuration
from .config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the main application
app = FastAPI(
    title="Physical AI & Humanoid Robotics API",
    description="API for the interactive textbook with AI-powered Q&A capabilities",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(auth_router)
app.include_router(status_router)

@app.get("/")
async def root():
    """
    Root endpoint for API health check
    """
    return {
        "message": "Physical AI & Humanoid Robotics API - Docusaurus + Isaac Sim + vSLAM + RL",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "config_valid": len(config.validate()) == 0 if hasattr(config, 'validate') else True
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        # Check services availability
        services_healthy = {
            'gemini_api': config.GEMINI_API_KEY is not None,
            'qdrant_db': config.QDRANT_URL is not None and config.QDRANT_API_KEY is not None,
            'database': config.NEON_DB_URL is not None,
            'better_auth': config.BETTER_AUTH_SECRET is not None
        }
        
        # Overall health status
        overall_healthy = all(services_healthy.values())
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": services_healthy,
            "overall_health": overall_healthy
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup
    """
    logger.info("Starting up Physical AI & Humanoid Robotics API...")
    
    # Validate configuration
    if hasattr(config, 'validate'):
        validation_errors = config.validate()
        if validation_errors:
            logger.warning("Configuration validation warnings:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
    
    logger.info("API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup services on shutdown
    """
    logger.info("Shutting down Physical AI & Humanoid Robotics API...")

# If you need to include static files or other assets
import os
from fastapi.staticfiles import StaticFiles

# Check if build directory exists (for Docusaurus integration)
if os.path.exists("./build"):
    app.mount("/", StaticFiles(directory="./build", html=True), name="static")
else:
    logger.info("Build directory not found. Serving only API endpoints.")