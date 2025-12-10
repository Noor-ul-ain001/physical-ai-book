"""
RAG-powered API for Physical AI Textbook Chatbot
Full implementation with Gemini and Qdrant
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "physical_ai_textbook"
TOP_K_RESULTS = 3

# Create FastAPI app
app = FastAPI(
    title="Physical AI Textbook API",
    description="RAG-powered chatbot API with Gemini and Qdrant",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
gemini_api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Check if RAG is available
rag_available = all([gemini_api_key, qdrant_url, qdrant_api_key])

if rag_available:
    try:
        genai.configure(api_key=gemini_api_key)
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        logger.info("✓ RAG services initialized (Gemini + Qdrant)")
    except Exception as e:
        logger.error(f"Failed to initialize RAG services: {e}")
        rag_available = False
else:
    logger.warning("RAG not configured - using fallback responses")
    qdrant_client = None

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

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Gemini"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def search_similar_content(query: str, limit: int = TOP_K_RESULTS):
    """Search for similar content in Qdrant"""
    try:
        # Generate query embedding
        query_embedding = get_embedding(query)

        if not query_embedding:
            return []

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                'title': result.payload.get('title', 'Unknown'),
                'content': result.payload.get('content', ''),
                'url': result.payload.get('url', '#'),
                'score': result.score,
            })

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def generate_rag_response(question: str, context: str, history: List[Message]) -> str:
    """Generate response using Gemini with RAG context"""
    try:
        # Build conversation history
        conversation_context = ""
        if history:
            for msg in history[-5:]:  # Last 5 messages
                role = "User" if msg.sender == "user" else "Assistant"
                conversation_context += f"{role}: {msg.text}\n\n"

        # Create prompt with RAG context
        prompt = f"""You are an expert AI assistant for a Physical AI and Humanoid Robotics textbook.
Your role is to help students understand complex concepts about robotics, AI, Isaac Sim, ROS2, vSLAM, and humanoid systems.

**Relevant content from the textbook:**
{context}

**Previous conversation:**
{conversation_context}

**User's question:** {question}

**Instructions:**
1. Provide a clear, accurate answer based PRIMARILY on the textbook content above
2. If the textbook content doesn't fully answer the question, acknowledge what you can't answer from the material
3. Use markdown formatting for better readability (bold, lists, code blocks, etc.)
4. Be concise but thorough
5. When relevant, mention related concepts the student should explore

**Answer:**"""

        # Generate response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I apologize, but I encountered an error generating a response. Please try again."

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Physical AI & Humanoid Robotics API",
        "version": "2.0.0",
        "status": "running",
        "rag_enabled": rag_available,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_available": rag_available,
        "gemini_configured": bool(gemini_api_key),
        "qdrant_configured": bool(qdrant_url and qdrant_api_key),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG
    Uses Qdrant vector search + Gemini for contextual responses
    """
    try:
        logger.info(f"Chat request: {request.message[:100]}...")

        if not rag_available:
            return ChatResponse(
                response="⚠️ RAG services are not configured. Please set GEMINI_API_KEY, QDRANT_URL, and QDRANT_API_KEY in your .env file.\n\nFor now, please refer to the textbook chapters directly.",
                sources=[]
            )

        # Search for relevant content
        search_results = search_similar_content(request.message, limit=TOP_K_RESULTS)

        if not search_results:
            return ChatResponse(
                response="I couldn't find relevant content in the textbook for your question. Could you rephrase it or ask about a specific topic like ROS2, Isaac Sim, vSLAM, or reinforcement learning?",
                sources=[]
            )

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"### Source {i}: {result['title']}\n"
                f"{result['content']}\n"
                f"(Relevance: {result['score']:.2f})\n"
            )

        context = "\n".join(context_parts)

        # Generate response with RAG
        response_text = generate_rag_response(
            request.message,
            context,
            request.history
        )

        # Format sources
        sources = [
            Source(
                title=result['title'],
                url=result['url']
            )
            for result in search_results
        ]

        logger.info(f"✓ Generated response with {len(sources)} sources")

        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again."
        )

@app.get("/api/collection-info")
async def collection_info():
    """Get information about the Qdrant collection"""
    if not rag_available or not qdrant_client:
        return {"error": "Qdrant not configured"}

    try:
        collection = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": collection.vectors_count,
            "indexed": collection.points_count,
            "status": "ready"
        }
    except Exception as e:
        return {
            "collection_name": COLLECTION_NAME,
            "status": "not_indexed",
            "error": str(e),
            "message": "Run 'python scripts/index_content.py' to index the content"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
