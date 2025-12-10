# Physical AI Textbook API

FastAPI backend for the RAG-powered chatbot.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   Make sure the following are set in the root `.env` file:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `QDRANT_URL`: Your Qdrant Cloud URL
   - `QDRANT_API_KEY`: Your Qdrant API key

3. **Run the server**:
   ```bash
   cd api
   python main.py
   ```

   Or using uvicorn:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

## API Endpoints

### `GET /`
Health check and status endpoint.

### `POST /api/chat`
Chat endpoint for RAG-powered responses.

**Request body**:
```json
{
  "message": "What is Isaac Sim?",
  "history": [
    {
      "role": "user",
      "content": "Previous message"
    }
  ]
}
```

**Response**:
```json
{
  "response": "Isaac Sim is...",
  "sources": [
    {
      "title": "Module Title",
      "snippet": "Relevant content..."
    }
  ]
}
```

### `GET /api/health`
Detailed health check with connection status.

## Development

The API runs on port 8000 by default. The Docusaurus dev server (port 3000) automatically proxies `/api/*` requests to the FastAPI backend.

## Production Deployment

For production, you can deploy this FastAPI backend to:
- Vercel (using Vercel Functions)
- AWS Lambda
- Google Cloud Run
- Self-hosted server

Make sure to update the API URL in the frontend chatbot component for production.
