# Scripts Documentation

This directory contains utility scripts for setting up and managing the RAG chatbot.

## Scripts Overview

### 1. `setup_rag.py`
**Interactive configuration helper**

Guides you through setting up API keys and environment variables.

**Usage:**
```bash
python scripts/setup_rag.py
```

**What it does:**
- Checks for existing `.env` file
- Prompts for Gemini API key
- Prompts for Qdrant Cloud credentials
- Saves configuration to `.env`
- Shows next steps

**When to use:**
- First-time setup
- Updating API credentials
- Troubleshooting configuration

---

### 2. `index_content.py`
**Content indexing script**

Processes textbook content and uploads it to Qdrant for RAG.

**Usage:**
```bash
python scripts/index_content.py
```

**What it does:**
1. Reads all MDX/MD files from `docs/` directory
2. Cleans and processes content (removes JSX, HTML comments)
3. Splits content into chunks (~1000 characters each)
4. Generates embeddings using Gemini API
5. Uploads vectors to Qdrant Cloud
6. Creates searchable knowledge base

**Configuration:**
Edit these variables in the script:
- `CHUNK_SIZE`: Characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `BATCH_SIZE`: Upload batch size (default: 10)
- `COLLECTION_NAME`: Qdrant collection name

**Performance:**
- **Time**: ~5-10 minutes for 23 chapters
- **API calls**: ~2-3 per chunk (embedding + upload)
- **Rate limiting**: Built-in delays to respect API limits
- **Cost**: Free tier sufficient for most use cases

**Re-indexing:**
Run again when:
- Adding new chapters
- Updating existing content
- Changing chunk parameters
- Switching Qdrant clusters

**Output example:**
```
============================================================
Starting Content Indexing
============================================================

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

---

## Common Tasks

### First-Time Setup
```bash
# 1. Configure environment
python scripts/setup_rag.py

# 2. Index content
python scripts/index_content.py
```

### Update Content
```bash
# After adding/modifying docs
python scripts/index_content.py
```

### Check Configuration
```bash
# View current settings
cat ../.env

# Or re-run setup
python scripts/setup_rag.py
```

## Troubleshooting

### "No module named 'google.generativeai'"
```bash
cd api
pip install -r requirements.txt
```

### "Connection error" or "Invalid API key"
1. Check `.env` file exists
2. Verify API keys are correct
3. Test credentials:
```bash
# Test Gemini
curl -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}' \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_KEY"

# Test Qdrant
curl -H "api-key: YOUR_KEY" \
  "https://your-cluster.qdrant.tech/collections"
```

### Indexing stuck or slow
- Check internet connection
- Wait for rate limits to reset (60 requests/minute)
- Script auto-retries with backoff

### "Collection already exists"
The script automatically recreates the collection. This is normal.

## Advanced Usage

### Custom Chunk Size
Edit `index_content.py`:
```python
CHUNK_SIZE = 1500  # Larger chunks (more context)
CHUNK_OVERLAP = 300  # More overlap
```

### Selective Indexing
Modify the glob pattern in `index_content.py`:
```python
# Index only specific module
mdx_files = list(DOCS_DIR.glob('module1-ros2/**/*.mdx'))
```

### Multiple Collections
Create separate collections for different content:
```python
COLLECTION_NAME = "advanced_robotics"  # Change name
```

## API Reference

### Collection Info Endpoint
```bash
curl http://localhost:8000/api/collection-info
```

Response:
```json
{
  "collection_name": "physical_ai_textbook",
  "vectors_count": 156,
  "indexed": 156,
  "status": "ready"
}
```

## Performance Optimization

### Faster Indexing
1. Use paid Gemini API tier (higher rate limits)
2. Reduce `CHUNK_SIZE` for fewer API calls
3. Batch process in parallel (advanced)

### Better Search Results
1. Increase `TOP_K_RESULTS` in `rag_main.py`
2. Fine-tune chunk size for your content
3. Add metadata filtering (advanced)

### Cost Optimization
1. Use larger chunks (fewer embeddings)
2. Cache frequent queries
3. Implement query deduplication

## Files Created

After running these scripts, you'll have:

```
.env                          # API credentials
docs/                         # Your content (unchanged)
  └── **/*.mdx
scripts/
  ├── setup_rag.py           # This helped configure
  ├── index_content.py       # This indexed content
  └── README.md              # This file
```

And in Qdrant Cloud:
```
Collection: physical_ai_textbook
├── Vector 1 (embedding + metadata)
├── Vector 2 (embedding + metadata)
├── ...
└── Vector 156 (embedding + metadata)
```

## Next Steps

After successful indexing:

1. **Start Backend:**
   ```bash
   cd api
   python rag_main.py
   ```

2. **Test Queries:**
   Visit http://localhost:3000 and chat!

3. **Monitor Usage:**
   - Check Qdrant dashboard
   - Review Gemini API quota
   - Monitor response times

4. **Iterate:**
   - Add more content
   - Refine prompts
   - Customize responses

## Support

For issues:
1. Check script logs for errors
2. Verify `.env` configuration
3. Test API connections
4. Review [Gemini Docs](https://ai.google.dev/docs)
5. Review [Qdrant Docs](https://qdrant.tech/documentation/)
