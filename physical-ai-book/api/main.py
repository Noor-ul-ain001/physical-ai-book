# api/main.py
# FastAPI backend for humanoid robot AI services

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import asyncio
import logging
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

app = FastAPI(title="Physical AI & Humanoid Robotics API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify proper origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    userId: Optional[str] = None
    conversationId: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    response: str
    sources: List[Dict]
    conversationId: str

class HighlightChatRequest(BaseModel):
    selectedText: str
    question: str
    userId: Optional[str] = None
    conversationId: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatMessage):
    """
    Chat endpoint that queries the textbook content using RAG
    """
    try:
        # Retrieve relevant context from vector store
        search_results = qdrant_client.search(
            collection_name="physical-ai-book",
            query_text=request.message,
            limit=5  # Retrieve top 5 relevant chunks
        )
        
        # Format context from search results
        context_parts = []
        sources = []
        
        for result in search_results:
            if hasattr(result, 'payload') and 'content' in result.payload:
                context_parts.append(result.payload['content'])
                
                # Add source reference with URL and title
                source = {
                    "title": result.payload.get('title', ''),
                    "url": result.payload.get('source_url', ''),
                    "snippet": result.payload['content'][:200] + "..."  # First 200 chars
                }
                sources.append(source)
        
        # Combine context for RAG
        combined_context = "\n\n".join(context_parts)
        
        # Construct the prompt for Gemini
        full_prompt = f"""
        You are an expert AI assistant for the Physical AI & Humanoid Robotics interactive textbook. 
        Use the following context to answer the user's question. 
        If the answer is not in the context, say so clearly.
        
        Context:
        {combined_context}
        
        User question: {request.message}
        
        Answer:
        """
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        response_text = response.text if response else "I couldn't generate a response. Please try again."
        
        # Generate conversation ID if not provided
        import uuid
        conversation_id = request.conversationId or str(uuid.uuid4())
        
        # Create response
        chat_response = ChatResponse(
            id=str(uuid.uuid4()),
            response=response_text,
            sources=sources,
            conversationId=conversation_id
        )
        
        return chat_response
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.post("/api/highlight-chat", response_model=ChatResponse)
async def highlight_chat_endpoint(request: HighlightChatRequest):
    """
    Chat endpoint that uses only the selected/highlighted text as context
    """
    try:
        # For highlight chat, we only use the selected text as context
        selected_text = request.selectedText
        question = request.question
        
        # Construct the prompt with only the selected text as context
        full_prompt = f"""
        You are an expert AI assistant for the Physical AI & Humanoid Robotics interactive textbook. 
        Answer the user's question based ONLY on the following selected text:
        
        Selected text: {selected_text}
        
        User question about the selected text: {question}
        
        Answer:
        """
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        response_text = response.text if response else "I couldn't generate a response. Please try again."
        
        # Generate conversation ID if not provided
        import uuid
        conversation_id = request.conversationId or str(uuid.uuid4())
        
        # For highlight chat, we still return sources but they'll just be the selected text
        sources = [{
            "title": "Selected Text",
            "url": "#highlight-selection",  # Placeholder for highlight context
            "snippet": selected_text[:200] + "..."
        }]
        
        # Create response
        chat_response = ChatResponse(
            id=str(uuid.uuid4()),
            response=response_text,
            sources=sources,
            conversationId=conversation_id
        )
        
        return chat_response
        
    except Exception as e:
        logging.error(f"Highlight chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Highlight chat processing error: {str(e)}")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            # Process message through RAG pipeline
            # This would be similar to the POST endpoint but with streaming
            # Implementation depends on specific requirements
            
            # For now, send a simple acknowledgment
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)