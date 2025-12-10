# api/routers/chat_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import asyncio
import json
import uuid
from datetime import datetime

# Import configuration
from ..config import config

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
chat_router = APIRouter(prefix="/chat", tags=["chat"])

# Models
class ChatRequest(BaseModel):
    message: str
    page_context: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    response: str
    sources: List[Dict]
    conversation_id: str
    timestamp: str

class RAGQueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    confidence: float
    query_time_ms: float

async def build_context_from_documentation(query: str, page_context: Optional[str] = None) -> List[Dict]:
    """
    Build context from textbook documentation using RAG
    """
    # In a real implementation, this would connect to Qdrant to retrieve relevant content
    # For this demonstration, we'll return mock context
    import random

    modules = [
        "Module 1: ROS2 Fundamentals",
        "Module 2: Digital Twin & Simulation",
        "Module 3: Isaac Sim & vSLAM",
        "Module 4: Vision-Language-Action Models"
    ]

    sections = [
        "ROS2 Overview",
        "Nodes, Topics, Services",
        "URDF for Humanoids",
        "Gazebo Simulation",
        "Isaac Sim Basics",
        "vSLAM Implementation",
        "Reinforcement Learning",
        "Personalized Learning"
    ]

    mock_contexts = []
    for i in range(3):  # Return 3 mock contexts
        module = random.choice(modules)
        section = random.choice(sections)

        mock_contexts.append({
            'section': section,
            'module': module,
            'title': f"Relevant content for: {query[:30]}...",
            'url': f"/docs/{module.lower().replace(' ', '-')}/{section.lower().replace(' ', '-')}",
            'content': f"Here is some relevant content from {module}, section {section} that relates to your query: {query}. This would be actual textbook content in the full implementation.",
            'score': 0.8 - (i * 0.1)  # Decreasing relevance scores
        })

    return mock_contexts

async def build_sources_list(context: List[Dict]) -> List[Dict]:
    """
    Build sources list from context for citation in responses
    """
    sources = []
    for item in context:
        source = {
            "title": item.get('title', 'Unknown Source'),
            "url": item.get('url', '#'),
            "module": item.get('module', 'Unknown'),
            "section": item.get('section', 'Unknown'),
            "relevance_score": item.get('score', 0.0),
            "snippet": item.get('content', '')[:200] + "..."  # First 200 chars
        }
        sources.append(source)

    return sources

def construct_prompt_for_gemini(query: str, context: List[Dict], user_context: Optional[Dict] = None) -> str:
    """
    Construct prompt for Gemini with context and user information
    """
    # Build context string
    context_str = ""
    for item in context:
        context_str += f"Module: {item['module']}\n"
        context_str += f"Section: {item['section']}\n"
        context_str += f"Content: {item['content']}\n\n"

    # Build user context string
    user_context_str = ""
    if user_context:
        user_context_str = f"\nUser Profile: {json.dumps(user_context, indent=2)}\n"
        user_context_str += "Please personalize your response based on the user's background and goals.\n"

    # Construct full prompt
    prompt = f"""
    You are an expert AI assistant for the Physical AI & Humanoid Robotics interactive textbook.
    Please answer the user's question using the provided context from the textbook.

    CONTEXT FROM TEXTBOOK:
    {context_str}

    USER'S QUESTION: {query}
    {user_context_str}

    INSTRUCTIONS:
    1. Answer using only the context provided from the textbook
    2. If the answer is not in the context, clearly state this limitation
    3. Provide specific citations to relevant modules and sections
    4. If user context is provided, personalize the response appropriately
    5. Keep responses concise but informative
    6. For technical questions, provide code examples when appropriate
    7. If asked about concepts not in the context, suggest related topics that might be covered elsewhere

    ANSWER:
    """

    return prompt

@chat_router.post("/", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Handle chat requests with RAG-powered responses
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Build context from documentation
        context = await build_context_from_documentation(request.message, request.page_context)

        # Construct prompt for Gemini
        full_prompt = construct_prompt_for_gemini(request.message, context, request.user_context)

        # In a real implementation, we would use the actual Gemini API
        if config.GEMINI_API_KEY:
            # Use actual Gemini API
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

            response = model.generate_content(full_prompt)
            answer = response.text if response else "I couldn't generate a response. Please try again."
        else:
            # Use mock response for demonstration
            answer = f"This is a mock response to your message: '{request.message}'. In the actual implementation, this would connect to the Gemini API using your GEMINI_API_KEY environment variable."

        # Create sources list (would come from RAG system)
        sources = await build_sources_list(context)

        # Calculate processing time
        query_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        logger.info(f"Chat request processed in {query_time_ms:.2f}ms")

        # Generate response
        response_obj = ChatResponse(
            id=str(uuid.uuid4()),
            response=answer,
            sources=sources,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        )

        return response_obj

    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@chat_router.post("/highlight", response_model=ChatResponse)
async def handle_highlight_chat(request: ChatRequest):
    """
    Handle chat requests specifically about highlighted/selected text only
    """
    start_time = asyncio.get_event_loop().time()
    selected_text = request.message  # For this endpoint, the message is the selected text

    try:
        # Generate response based specifically on the selected text
        full_prompt = f"""
        You are an expert AI assistant for the Physical AI & Humanoid Robotics interactive textbook.
        The user has highlighted the following text and wants to know more about it:

        SELECTED TEXT: {selected_text}

        USER'S QUESTION ABOUT THE SELECTED TEXT: {request.page_context or 'Explain this concept in simpler terms'}

        INSTRUCTIONS:
        1. Answer specifically about the selected text content
        2. Provide a detailed explanation of the concept
        3. If there are related concepts in the textbook that connect to this, mention them
        4. Do not introduce unrelated content
        5. Cite the source section if known
        6. If the text is code, explain what it does and how it works

        ANSWER:
        """

        # Generate response
        if config.GEMINI_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

            response = model.generate_content(full_prompt)
            answer = response.text if response else "I couldn't generate a response about the selected text."
        else:
            answer = f"You selected: '{selected_text[:50]}...'. This is where the AI would provide specific insights about the highlighted content. In the full implementation, this would use the Gemini API."

        # Calculate processing time
        query_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Create response object
        response_obj = ChatResponse(
            id=str(uuid.uuid4()),
            response=answer,
            sources=[{
                "title": "Selected Text",
                "url": request.page_context or "#",
                "module": "Context",
                "section": "Highlighted Content",
                "relevance_score": 1.0,
                "snippet": selected_text[:200] + "..."
            }],
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        )

        return response_obj

    except Exception as e:
        logger.error(f"Highlight chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Highlight chat processing failed: {str(e)}")

# Add a health check endpoint for the chat router
@chat_router.get("/health")
async def chat_health_check():
    """
    Health check for the chat system
    """
    return {
        "status": "healthy",
        "model_configured": config.GEMINI_API_KEY is not None,
        "timestamp": datetime.now().isoformat()
    }

# Add a test endpoint to verify functionality
@chat_router.post("/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    Test endpoint to verify chat functionality
    """
    import time

    start_time = time.time()

    # Simple test response
    test_response = f"This is a test response to your message: '{request.message}'. The system is functioning correctly."

    sources = [{
        "title": "Test Response",
        "url": "/docs/test",
        "module": "Test Module",
        "section": "Test Section",
        "relevance_score": 1.0,
        "snippet": "This is a simple test response to verify system functionality."
    }]

    response_obj = ChatResponse(
        id=str(uuid.uuid4()),
        response=test_response,
        sources=sources,
        conversation_id=request.conversation_id or str(uuid.uuid4()),
        timestamp=datetime.now().isoformat()
    )

    return response_obj