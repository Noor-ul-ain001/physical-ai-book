# api/config.py

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Application configuration settings for the Physical AI & Humanoid Robotics project
    """
    # Gemini API Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
    
    # Qdrant (vector database) Configuration
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = "physical-ai-book"
    
    # Database Configuration (Neon Serverless Postgres)
    NEON_DB_URL: Optional[str] = os.getenv("NEON_DB_URL")
    
    # Better-Auth Configuration
    BETTER_AUTH_SECRET: Optional[str] = os.getenv("BETTER_AUTH_SECRET")
    
    # Application Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # vSLAM/RAG Settings
    RAG_TOP_K: int = 5  # Number of results to retrieve
    RAG_SEARCH_THRESHOLD: float = 0.3  # Similarity threshold for relevance
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "embedding-001"  # Gemini embedding model
    EMBEDDING_DIMENSIONS: int = 768  # Dimensions of Gemini embeddings
    
    # Chat Settings
    CHAT_MAX_HISTORY: int = 20  # Max conversation history items
    CHAT_TIMEOUT_SECONDS: int = 30  # Timeout for chat responses
    
    # File Paths
    DOCS_PATH: str = os.getenv("DOCS_PATH", "../docs")
    TEMP_PATH: str = os.getenv("TEMP_PATH", "./temp")
    
    def validate(self):
        """
        Validate configuration settings and return any issues
        """
        errors = []
        
        if not self.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY not set - using mock responses")
            
        if not self.QDRANT_URL or not self.QDRANT_API_KEY:
            errors.append("QDRANT_URL or QDRANT_API_KEY not set - using mock RAG")
            
        if not self.NEON_DB_URL:
            errors.append("NEON_DB_URL not set - using mock storage")
            
        if not self.BETTER_AUTH_SECRET:
            errors.append("BETTER_AUTH_SECRET not set - authentication may not work properly")
        
        return errors

# Initialize global configuration
config = Config()

# Validate configuration at startup
config_validation_errors = config.validate()
if config_validation_errors:
    print("Configuration validation notes (non-critical):")
    for error in config_validation_errors:
        print(f"  - {error}")
    print("Using mock implementations for missing services where possible.")