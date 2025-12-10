#!/usr/bin/env python3
"""
Content Indexing Script for RAG Chatbot
Indexes all textbook content into Qdrant vector database
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import re
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import time

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = Path(__file__).parent.parent / "docs"
COLLECTION_NAME = "physical_ai_textbook"
BATCH_SIZE = 10
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

class ContentIndexer:
    def __init__(self):
        """Initialize the indexer with Gemini and Qdrant clients"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not all([self.gemini_api_key, self.qdrant_url, self.qdrant_api_key]):
            raise ValueError(
                "Missing required environment variables. Please set:\n"
                "- GEMINI_API_KEY\n"
                "- QDRANT_URL\n"
                "- QDRANT_API_KEY"
            )

        # Initialize Gemini
        genai.configure(api_key=self.gemini_api_key)

        # Initialize Qdrant
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        print("✓ Connected to Gemini API")
        print("✓ Connected to Qdrant")

    def extract_frontmatter(self, content: str) -> tuple[Dict, str]:
        """Extract frontmatter and content from MDX file"""
        frontmatter = {}
        main_content = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Parse frontmatter
                fm_lines = parts[1].strip().split('\n')
                for line in fm_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip()

                main_content = parts[2].strip()

        return frontmatter, main_content

    def clean_content(self, content: str) -> str:
        """Clean MDX content - remove JSX components but keep text"""
        # Remove import statements
        content = re.sub(r'^import\s+.*?from\s+[\'"].*?[\'"];?\s*$', '', content, flags=re.MULTILINE)

        # Remove JSX component tags but keep content inside
        content = re.sub(r'<[A-Z][a-zA-Z]*[^>]*>', '', content)
        content = re.sub(r'</[A-Z][a-zA-Z]*>', '', content)

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Remove code block language identifiers
        content = re.sub(r'```[a-z]*\n', '```\n', content)

        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    def chunk_content(self, content: str, title: str) -> List[Dict]:
        """Split content into chunks with overlap"""
        chunks = []

        # Split by paragraphs first
        paragraphs = content.split('\n\n')

        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) < CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'id': chunk_id,
                        'content': current_chunk.strip(),
                        'title': title,
                    })
                    chunk_id += 1

                # Start new chunk with overlap
                current_chunk = para + "\n\n"

        # Add the last chunk
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'content': current_chunk.strip(),
                'title': title,
            })

        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def index_file(self, file_path: Path) -> int:
        """Index a single MDX file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter and clean content
            frontmatter, main_content = self.extract_frontmatter(content)
            clean_text = self.clean_content(main_content)

            # Get title
            title = frontmatter.get('title', file_path.stem.replace('-', ' ').title())

            # Get relative path for URL
            rel_path = file_path.relative_to(DOCS_DIR)
            url_path = '/' + str(rel_path).replace('\\', '/').replace('.mdx', '').replace('.md', '')

            # Chunk the content
            chunks = self.chunk_content(clean_text, title)

            indexed_count = 0
            points = []

            for chunk in chunks:
                # Generate embedding
                embedding = self.get_embedding(chunk['content'])

                if embedding:
                    # Create point for Qdrant
                    point_id = hash(f"{file_path}_{chunk['id']}") % (10 ** 8)

                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'title': title,
                            'content': chunk['content'],
                            'file_path': str(file_path),
                            'url': url_path,
                            'chunk_id': chunk['id'],
                            'source': f"{title} (chunk {chunk['id']})"
                        }
                    )

                    points.append(point)
                    indexed_count += 1

                # Rate limiting
                time.sleep(0.1)

            # Batch upload to Qdrant
            if points:
                self.qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )

            return indexed_count

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return 0

    def create_collection(self):
        """Create or recreate the Qdrant collection"""
        try:
            # Delete existing collection if it exists
            try:
                self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
                print(f"✓ Deleted existing collection: {COLLECTION_NAME}")
            except:
                pass

            # Create new collection
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,  # Gemini embedding-001 dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created new collection: {COLLECTION_NAME}")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def index_all_content(self):
        """Index all MDX files in the docs directory"""
        print(f"\n{'='*60}")
        print("Starting Content Indexing")
        print(f"{'='*60}\n")

        # Create collection
        self.create_collection()

        # Find all MDX and MD files
        mdx_files = list(DOCS_DIR.glob('**/*.mdx'))
        md_files = list(DOCS_DIR.glob('**/*.md'))
        all_files = mdx_files + md_files

        print(f"Found {len(all_files)} files to index\n")

        total_chunks = 0
        successful_files = 0

        for i, file_path in enumerate(all_files, 1):
            rel_path = file_path.relative_to(DOCS_DIR)
            print(f"[{i}/{len(all_files)}] Indexing: {rel_path}")

            chunks_indexed = self.index_file(file_path)

            if chunks_indexed > 0:
                successful_files += 1
                total_chunks += chunks_indexed
                print(f"  ✓ Indexed {chunks_indexed} chunks")
            else:
                print(f"  ✗ Failed to index")

            print()

        print(f"\n{'='*60}")
        print("Indexing Complete!")
        print(f"{'='*60}")
        print(f"Files processed: {successful_files}/{len(all_files)}")
        print(f"Total chunks indexed: {total_chunks}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"{'='*60}\n")

def main():
    """Main function"""
    try:
        indexer = ContentIndexer()
        indexer.index_all_content()

        print("\n✓ All content has been indexed successfully!")
        print("You can now use the RAG chatbot with full textbook content.\n")

    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
