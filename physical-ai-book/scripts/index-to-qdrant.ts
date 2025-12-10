// scripts/index-to-qdrant.ts
// Indexing script to parse all MDX files and create embeddings for RAG system

import fs from 'fs';
import path from 'path';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Document } from 'langchain/document';

// Configuration
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const COLLECTION_NAME = 'physical-ai-book';

if (!GEMINI_API_KEY || !QDRANT_URL || !QDRANT_API_KEY) {
  throw new Error('Missing required environment variables for indexing');
}

// Initialize clients
const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: 'embedding-001',
  apiKey: GEMINI_API_KEY,
});

const client = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY,
});

/**
 * Parse MDX content and extract text
 */
function parseMDXContent(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Remove frontmatter if it exists
  let textContent = content;
  const frontmatterRegex = /^---[\s\S]*?---/;
  const frontmatterMatch = content.match(frontmatterRegex);
  if (frontmatterMatch) {
    textContent = content.replace(frontmatterRegex, '');
  }
  
  // Remove JSX/TSX components and get just the text
  // This is a simple approach - in practice, you might want to parse MDX properly
  // For now, just return the content and let the text splitter handle it

  // Remove common MDX/JSX syntax but keep meaningful content
  let cleanContent = textContent
    .replace(/\{\{[^}]+\}\}/g, '') // Remove handlebars-style templates
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks (they'll be handled separately)
    .replace(/`[^`]*`/g, '') // Remove inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove markdown links, keep text
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '') // Remove image tags
    .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
    .replace(/\*([^*]+)\*/g, '$1') // Remove italic
    .replace(/~~([^~]+)~~/g, '$1'); // Remove strikethrough

  return cleanContent.trim();
}

/**
 * Extract content from all MDX files in docs directory
 */
function extractAllContent(docsPath) {
  const documents = [];
  
  // Walk through docs directory and collect all MDX files
  function walkDirectory(dir) {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        walkDirectory(filePath);
      } else if (path.extname(file) === '.mdx') {
        const content = parseMDXContent(filePath);
        const relativePath = path.relative(docsPath, filePath).replace(/\\/g, '/');
        const url = `/docs/${relativePath.replace('.mdx', '')}`;
        
        // Determine module and title from path
        const pathSegments = relativePath.split('/');
        const moduleName = pathSegments[0];
        const fileName = pathSegments[pathSegments.length - 1];
        const title = fileName.replace('.mdx', '').replace(/-/g, ' ');
        
        documents.push({
          pageContent: content,
          metadata: {
            source_url: url,
            module: moduleName,
            title: title,
            filepath: filePath
          }
        });
      }
    }
  }
  
  walkDirectory(docsPath);
  return documents;
}

/**
 * Create and index documents in Qdrant
 */
async function indexDocuments(documents) {
  console.log(`Processing ${documents.length} documents...`);
  
  // Split documents into chunks
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  
  const splitDocuments = await textSplitter.splitDocuments(documents);
  console.log(`Created ${splitDocuments.length} text chunks`);
  
  // Create vectors and store in Qdrant
  const points = [];
  
  for (let i = 0; i < splitDocuments.length; i++) {
    const doc = splitDocuments[i];
    const embedding = await embeddings.embedQuery(doc.pageContent);
    
    points.push({
      id: `${i}`, // Use index as ID (in production, use UUID)
      vector: embedding,
      payload: {
        content: doc.pageContent,
        source_url: doc.metadata.source_url,
        module: doc.metadata.module,
        title: doc.metadata.title,
        filepath: doc.metadata.filepath,
        position: i
      }
    });
    
    if (i % 100 === 0) {
      console.log(`Processed ${i}/${splitDocuments.length} chunks...`);
    }
  }
  
  // Create collection if it doesn't exist
  try {
    await client.getCollection(COLLECTION_NAME);
    console.log(`Collection '${COLLECTION_NAME}' already exists`);
  } catch (error) {
    console.log(`Creating collection '${COLLECTION_NAME}'...`);
    await client.createCollection(COLLECTION_NAME, {
      vectors: {
        size: 768, // Size of Gemini embedding-001 vectors
        distance: 'Cosine',
      },
    });
  }
  
  // Upload vectors to Qdrant
  console.log('Uploading vectors to Qdrant...');
  await client.upsert(COLLECTION_NAME, {
    points: points,
  });
  
  console.log(`Successfully indexed ${points.length} chunks to '${COLLECTION_NAME}' collection`);
}

/**
 * Main indexing function
 */
async function main() {
  const docsPath = path.join(__dirname, '../docs');
  
  if (!fs.existsSync(docsPath)) {
    console.error(`Docs directory does not exist: ${docsPath}`);
    process.exit(1);
  }
  
  console.log('Starting content indexing process...');
  
  try {
    // Extract content from all MDX files
    const documents = extractAllContent(docsPath);
    
    if (documents.length === 0) {
      console.warn('No MDX files found in docs directory');
      return;
    }
    
    // Index documents in Qdrant
    await indexDocuments(documents);
    
    console.log('Indexing completed successfully!');
  } catch (error) {
    console.error('Indexing failed:', error);
    process.exit(1);
  }
}

// Run the main function if this file is executed directly
if (require.main === module) {
  main();
}

export { extractAllContent, indexDocuments };