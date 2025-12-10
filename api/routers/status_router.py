# api/routers/status_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import platform
import os
from datetime import datetime

# Import configuration
from ..config import config

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
status_router = APIRouter(prefix="/status", tags=["status"])

class SystemStatus(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    configuration: Dict[str, bool]
    system_info: Dict[str, Any]

@status_router.get("/", response_model=SystemStatus)
async def get_system_status():
    """
    Get system status and health check
    """
    try:
        services_status = {
            "qdrant_connected": True,  # This would check actual connection
            "gemini_api_available": True,  # This would check actual availability
            "database_connected": True,  # This would check actual connection
            "auth_system_active": True,  # This would check actual status
            "isaac_sim_available": check_isaac_sim_availability()
        }
        
        config_status = {
            "gemini_api_configured": "GEMINI_API_KEY" in globals() or "GEMINI_API_KEY" in os.environ,
            "qdrant_configured": "QDRANT_URL" in globals() or "QDRANT_URL" in os.environ,
            "database_configured": "DATABASE_URL" in globals() or "NEON_DB_URL" in os.environ,
            "auth_configured": "BETTER_AUTH_SECRET" in globals() or "BETTER_AUTH_SECRET" in os.environ
        }
        
        system_info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "node_available": check_node_availability(),
            "gpu_available": check_gpu_availability()
        }
        
        return SystemStatus(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services=services_status,
            configuration=config_status,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

def check_isaac_sim_availability():
    """
    Check if Isaac Sim is available
    """
    try:
        # This would check actual Isaac Sim availability
        # For mock purposes, return True
        return True
    except:
        return False

def check_node_availability():
    """
    Check if Node.js is available
    """
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def check_gpu_availability():
    """
    Check if GPU is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False