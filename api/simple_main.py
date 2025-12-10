import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Physical AI Textbook API",
    description="RAG-powered chatbot API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    id: int
    text: str
    sender: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Source]] = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Physical AI & Humanoid Robotics API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - returns responses based on textbook content
    In production, this would use Gemini + Qdrant for RAG
    """
    try:
        logger.info(f"Received chat request: {request.message}")

        # For now, return intelligent responses based on keywords
        # In production, this would use RAG with Gemini and Qdrant
        user_message = request.message.lower()

        # Sample responses based on content
        if "ros" in user_message or "ros2" in user_message:
            response_text = """**ROS2 (Robot Operating System 2)** is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.

**Key features of ROS2:**
- **Distributed architecture**: Nodes communicate via publish-subscribe messaging
- **Real-time capabilities**: Support for real-time systems with DDS middleware
- **Platform agnostic**: Works on Linux, Windows, and macOS
- **Language support**: Python (rclpy) and C++ (rclcpp) client libraries

**In our textbook context**, we use ROS2 to control the humanoid robot, manage sensor data, and coordinate between simulation and real hardware."""
            sources = [
                Source(title="Module 1: ROS2 Overview", url="/docs/module1-ros2/overview"),
                Source(title="ROS2 Nodes, Topics & Services", url="/docs/module1-ros2/nodes-topics-services")
            ]

        elif "isaac" in user_message or "simulation" in user_message:
            response_text = """**Isaac Sim** is NVIDIA's robotics simulation platform built on the Omniverse platform. It provides physics-accurate, photorealistic virtual environments for training and testing robots.

**Key capabilities:**
- **Physics simulation**: Accurate dynamics using PhysX
- **Sensor simulation**: Cameras, LiDAR, IMU, and more
- **ROS2 integration**: Direct communication with ROS2 ecosystem
- **Synthetic data generation**: For training vision models
- **Parallel environments**: For reinforcement learning

**For our humanoid robot**, we use Isaac Sim to create a digital twin, enabling safe testing of control algorithms before deployment to real hardware."""
            sources = [
                Source(title="Module 3: Isaac Sim Basics", url="/docs/module3-isaac/isaac-sim"),
                Source(title="Isaac Sim & ROS Integration", url="/docs/module3-isaac/isaac-ros")
            ]

        elif "vslam" in user_message or "slam" in user_message or "navigation" in user_message:
            response_text = """**Visual SLAM (vSLAM)** is a technique that allows robots to build maps of their environment and localize themselves within those maps using only camera input.

**How it works:**
1. **Feature extraction**: Detect keypoints in camera images
2. **Feature matching**: Track features across frames
3. **Pose estimation**: Calculate camera movement
4. **Map building**: Create 3D map of the environment
5. **Loop closure**: Recognize previously visited locations

**In our humanoid project**, we use vSLAM for autonomous navigation in indoor environments, enabling the robot to move safely without prior knowledge of the space."""
            sources = [
                Source(title="vSLAM & Navigation", url="/docs/module3-isaac/vslam-navigation"),
                Source(title="Module 3 Summary", url="/docs/module3-isaac/project-summary")
            ]

        elif "reinforcement" in user_message or "rl" in user_message or "learning" in user_message:
            response_text = """**Reinforcement Learning (RL)** is a machine learning paradigm where agents learn optimal behaviors through trial and error by maximizing cumulative rewards.

**Key concepts:**
- **Agent**: The robot learning to perform tasks
- **Environment**: Isaac Sim simulation
- **State**: Sensor readings and robot configuration
- **Action**: Joint commands and control inputs
- **Reward**: Feedback signal (e.g., staying upright, walking forward)

**For humanoid control**, we use RL to learn complex behaviors like:
- Bipedal locomotion
- Balance and stability
- Terrain adaptation
- Dynamic movement

The simulation allows us to train for thousands of hours in minutes using GPU acceleration."""
            sources = [
                Source(title="Reinforcement Learning Basics", url="/docs/module3-isaac/reinforcement-learning"),
                Source(title="Project Completion", url="/docs/module3-isaac/project-completion")
            ]

        elif "python" in user_message or "rclpy" in user_message:
            response_text = """**rclpy** is the Python client library for ROS2, allowing you to create ROS2 nodes and interact with the ROS2 ecosystem using Python.

**Basic structure of a ROS2 Python node:**
```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS2!'
        self.publisher_.publish(msg)

rclpy.init()
node = MinimalPublisher()
rclpy.spin(node)
```

**Key features**: Publishers, subscribers, services, actions, parameters, and timers all accessible through Python."""
            sources = [
                Source(title="rclpy Python Bridge", url="/docs/module1-ros2/rclpy-python-bridge"),
                Source(title="ROS2 Fundamentals", url="/docs/module1-ros2/overview")
            ]

        elif "urdf" in user_message or "robot model" in user_message:
            response_text = """**URDF (Unified Robot Description Format)** is an XML format for representing a robot model in ROS2. It defines the robot's physical structure, joints, links, sensors, and visual/collision geometries.

**URDF components:**
- **Links**: Rigid bodies (bones) of the robot
- **Joints**: Connections between links (revolute, prismatic, fixed)
- **Sensors**: Cameras, LiDAR, IMU specifications
- **Visual**: 3D meshes for visualization
- **Collision**: Simplified geometry for physics

**For our humanoid**, the URDF defines all 20+ degrees of freedom, mass properties, and sensor placements, enabling accurate simulation in both Gazebo and Isaac Sim."""
            sources = [
                Source(title="URDF for Humanoids", url="/docs/module1-ros2/urdf-humanoids"),
                Source(title="URDF & SDF Formats", url="/docs/module2-digital-twin/urdf-sdf")
            ]

        else:
            response_text = """I'm your AI assistant for the Physical AI & Humanoid Robotics textbook. I can help you understand:

- **ROS2 fundamentals**: Nodes, topics, services, and Python integration
- **Digital twin concepts**: Simulation, URDF modeling, and Gazebo
- **Isaac Sim**: NVIDIA's robotics simulation platform
- **Visual SLAM**: Navigation and mapping techniques
- **Reinforcement Learning**: Training humanoid control policies
- **Humanoid robotics**: From hardware to software integration

Please ask me a specific question about any of these topics!"""
            sources = [
                Source(title="Getting Started", url="/docs/intro/welcome"),
                Source(title="Module Overview", url="/docs/intro/foundations")
            ]

        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
