# RAG Chatbot Setup Guide

This guide will help you set up the full RAG (Retrieval-Augmented Generation) chatbot with Gemini and Qdrant.

## Prerequisites

1. **Python 3.11+** installed
2. **Node.js 18+** installed
3. **Google Cloud account** with Gemini API access
4. **Qdrant Cloud account** (free tier available)

## Step 1: Get API Keys

### Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### Qdrant Cloud Setup

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Sign up for a free account
3. Create a new cluster:
   - Choose a region close to you
   - Select "Free" tier (1GB storage)
4. Once created, get:
   - Cluster URL (looks like: `https://xyz-abc.qdrant.tech`)
   - API Key (from cluster settings)

## Step 2: Configure Environment Variables

Update the `.env` file in the project root:

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Configuration
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here

# Database Configuration (optional for now)
NEON_DB_URL=postgresql://user:password@host/dbname

# Auth Configuration (optional for now)
BETTER_AUTH_SECRET=your_secret_here

# Isaac Sim Configuration (optional)
ISAAC_SIM_PATH=/path/to/isaac/sim
```

## Step 3: Install Python Dependencies

```bash
# Navigate to api directory
cd api

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Index Your Content

This step processes all textbook content and uploads it to Qdrant:

```bash
# From project root
python scripts/index_content.py
```

**Expected output:**
```
✓ Connected to Gemini API
✓ Connected to Qdrant
✓ Created new collection: physical_ai_textbook

Found 23 files to index

[1/23] Indexing: intro/welcome.mdx
  ✓ Indexed 3 chunks

[2/23] Indexing: intro/foundations.mdx
  ✓ Indexed 5 chunks

...

============================================================
Indexing Complete!
============================================================
Files processed: 23/23
Total chunks indexed: 156
Collection: physical_ai_textbook
============================================================
```

**⚠️ Important Notes:**
- Indexing may take 5-10 minutes depending on content size
- Gemini API has rate limits (60 requests/minute on free tier)
- The script includes automatic rate limiting
- You only need to run this once (or when content changes)

## Step 5: Start the RAG-Enabled Backend

```bash
# Stop the simple backend if running
# Press Ctrl+C in the terminal running simple_main.py

# Start RAG-enabled backend
cd api
python rag_main.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     ✓ RAG services initialized (Gemini + Qdrant)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 6: Test the Setup

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "rag_available": true,
  "gemini_configured": true,
  "qdrant_configured": true,
  "timestamp": "2025-12-10T..."
}
```

### Test 2: Collection Info
```bash
curl http://localhost:8000/api/collection-info
```

Expected response:
```json
{
  "collection_name": "physical_ai_textbook",
  "vectors_count": 156,
  "indexed": 156,
  "status": "ready"
}
```

### Test 3: Chat Query
Visit http://localhost:3000/ and click the chatbot icon. Try asking:
- "What is ROS2?"
- "Explain Isaac Sim"
- "How does vSLAM work?"

## Troubleshooting

### Issue: "RAG services are not configured"

**Solution:** Check that your `.env` file has valid API keys:
```bash
# Test Gemini API
curl -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
  https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_KEY

# Test Qdrant connection
curl -H "api-key: YOUR_QDRANT_KEY" \
  https://your-cluster.qdrant.tech/collections
```

### Issue: "Collection not found"

**Solution:** Run the indexing script:
```bash
python scripts/index_content.py
```

### Issue: Rate limit errors during indexing

**Solution:** The script includes rate limiting, but if you hit limits:
1. Wait a few minutes
2. Run the script again (it will skip already indexed content)
3. Or upgrade to Gemini API paid tier

### Issue: Embeddings returning None

**Solution:**
1. Check your Gemini API key is valid
2. Ensure you have API access enabled in Google Cloud Console
3. Check the logs for specific error messages

## File Structure

```
project/
├── .env                          # Your API keys (DO NOT commit!)
├── api/
│   ├── simple_main.py           # Fallback API (keyword-based)
│   ├── rag_main.py              # Full RAG API (Gemini + Qdrant)
│   └── requirements.txt         # Python dependencies
├── scripts/
│   └── index_content.py         # Content indexing script
├── docs/                        # Your textbook content
│   ├── intro/
│   ├── module1-ros2/
│   ├── module2-digital-twin/
│   └── module3-isaac/
└── src/
    ├── components/
    │   └── Chatbot.tsx          # Chatbot UI component
    └── theme/
        └── Root.tsx             # Integrates chatbot into all pages
```

## Cost Considerations

### Free Tier Limits

**Gemini API (Free):**
- 60 requests per minute
- 1,500 requests per day
- Sufficient for development and small projects

**Qdrant Cloud (Free):**
- 1GB storage
- Unlimited queries
- Sufficient for this textbook (~156 vectors)

### Estimated Costs (if exceeding free tier)

**Gemini API:**
- Embedding: $0.025 per 1K requests
- Generation: $0.05 per 1K requests
- For 100 daily users: ~$5-10/month

**Qdrant Cloud:**
- $10/month for 2GB cluster
- Only needed if expanding beyond 1GB

## Production Deployment

For production deployment:

1. **Environment Variables:** Use secure secret management
2. **Rate Limiting:** Add API rate limiting middleware
3. **Caching:** Cache frequent queries
4. **Monitoring:** Add logging and error tracking
5. **Scaling:** Consider Qdrant self-hosted for larger scale

## Support

If you encounter issues:

1. Check the logs in the terminal
2. Verify API keys are correct
3. Ensure all dependencies are installed
4. Check [Gemini API Docs](https://ai.google.dev/docs)
5. Check [Qdrant Docs](https://qdrant.tech/documentation/)

## Next Steps

Once RAG is working:

1. **Customize prompts** in `rag_main.py` for your use case
2. **Add user authentication** with Better-Auth
3. **Implement conversation memory** for multi-turn dialogs
4. **Add feedback collection** to improve responses
5. **Create admin dashboard** to monitor usage

Happy chatting! 🤖
