# api/routers/auth_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import uuid
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
auth_router = APIRouter(prefix="/auth", tags=["auth"])

class AuthRequest(BaseModel):
    email: str
    password: str
    hardware_gpu: Optional[bool] = False
    hardware_jetson: Optional[bool] = False
    hardware_robot: Optional[bool] = False
    python_experience: Optional[int] = 0
    ros_experience: Optional[str] = "none"  # "none", "ros1", "ros2", "both"
    linux_proficiency: Optional[int] = 0   # 0-10 scale
    rl_experience: Optional[int] = 0       # 0-10 scale
    primary_goal: Optional[str] = "learning"  # "learning", "building", "research"

class AuthResponse(BaseModel):
    success: bool
    user_id: str
    token: str
    message: str
    profile: Optional[Dict] = None

@auth_router.post("/register", response_model=AuthResponse)
async def register_user(auth_request: AuthRequest):
    """
    Register a new user with profile information
    """
    try:
        # In a real implementation, this would integrate with Better-Auth
        # and store profile information in Neon Postgres
        
        # Validate input
        if len(auth_request.email) < 5 or "@" not in auth_request.email:
            raise HTTPException(status_code=400, detail="Invalid email")
        
        if len(auth_request.password) < 8:
            raise HTTPException(status_code=400, detail="Password too short")
        
        # Mock registration
        user_id = str(uuid.uuid4())
        token = str(uuid.uuid4())
        
        profile = {
            "hardware_gpu": auth_request.hardware_gpu,
            "hardware_jetson": auth_request.hardware_jetson,
            "hardware_robot": auth_request.hardware_robot,
            "python_experience": auth_request.python_experience,
            "ros_experience": auth_request.ros_experience,
            "linux_proficiency": auth_request.linux_proficiency,
            "rl_experience": auth_request.rl_experience,
            "primary_goal": auth_request.primary_goal,
            "registration_date": datetime.now().isoformat()
        }
        
        return AuthResponse(
            success=True,
            user_id=user_id,
            token=token,
            message="Registration successful",
            profile=profile
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@auth_router.post("/login", response_model=AuthResponse)
async def login_user(auth_request: AuthRequest):
    """
    Login user with email/password
    """
    try:
        # In a real implementation, this would authenticate with Better-Auth
        # and retrieve profile information
        
        # Mock login (accept any email/password for demo)
        user_id = str(uuid.uuid4())
        token = str(uuid.uuid4())
        
        return AuthResponse(
            success=True,
            user_id=user_id,
            token=token,
            message="Login successful"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")