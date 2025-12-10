# Data Model: Physical AI & Humanoid Robotics Interactive Textbook

**Feature**: Physical AI & Humanoid Robotics Interactive Textbook  
**Date**: 2025-12-10

## Overview

This document defines the data models for the Physical AI & Humanoid Robotics Interactive Textbook project, including user data, content, and interaction tracking. The models are designed to support the required functionality while maintaining privacy and performance requirements.

## User Models

### User
- `id`: string (UUID) - Primary identifier
- `email`: string - User's email address
- `name`: string - User's full name
- `emailVerified`: datetime - When email was verified
- `image`: string (optional) - Profile picture URL (for OAuth)
- `createdAt`: datetime - Account creation timestamp
- `updatedAt`: datetime - Last update timestamp

### UserProfile
- `id`: string (UUID) - Primary identifier
- `userId`: string (UUID) - Foreign key to User
- `hardware_gpu`: boolean - Has RTX 4070+ GPU
- `hardware_jetson`: boolean - Owns a Jetson
- `hardware_robot`: boolean - Has a real robot
- `python_experience`: number (0-10) - Years with Python
- `ros_experience`: 'none' | 'ros1' | 'ros2' | 'both' - ROS experience level
- `linux_proficiency`: number (0-10) - Linux proficiency level
- `rl_experience`: number (0-10) - Experience with Reinforcement Learning/Isaac Sim
- `primary_goal`: 'learning' | 'building' | 'research' - Learning goal
- `skill_level`: 'beginner' | 'intermediate' | 'advanced' - Overall skill level (derived from other fields)
- `preferred_language`: string - Preferred language (default: 'en')
- `createdAt`: datetime - Profile creation timestamp
- `updatedAt`: datetime - Last update timestamp

## Content Models

### Chapter
- `id`: string (UUID) - Primary identifier
- `slug`: string - URL-friendly identifier (e.g., "intro/00-welcome", "module1-ros2/01-overview")
- `title`: string - Chapter title
- `module`: string - Module identifier (e.g., "intro", "module1-ros2", "module2-digital-twin")
- `order`: number - Order within the module
- `url`: string - Full URL path
- `content_summary`: string - Brief summary of content
- `createdAt`: datetime - Creation timestamp
- `updatedAt`: datetime - Last update timestamp

## Chat & Interaction Models

### Conversation
- `id`: string (UUID) - Primary identifier
- `userId`: string (UUID) - Foreign key to User (nullable for anonymous)
- `title`: string - Auto-generated title from first message
- `createdAt`: datetime - Creation timestamp
- `updatedAt`: datetime - Last update timestamp

### Message
- `id`: string (UUID) - Primary identifier
- `conversationId`: string (UUID) - Foreign key to Conversation
- `role`: 'user' | 'assistant' - Message sender
- `content`: string - Message content
- `sources`: array of SourceReference - Citations for assistant responses
- `createdAt`: datetime - Creation timestamp
- `updatedAt`: datetime - Last update timestamp

### SourceReference
- `id`: string (UUID) - Primary identifier
- `messageId`: string (UUID) - Foreign key to Message
- `chapterId`: string (UUID) - Reference to the chapter
- `url`: string - Direct URL to the referenced section
- `snippet`: string - Relevant text snippet
- `relevance_score`: number - How relevant this source was to the response (0-1)

## Vector Database Schema (Qdrant)

### TextChunk
- `id`: string (UUID) - Primary identifier
- `chapter_id`: string (UUID) - Reference to the chapter
- `content`: string - Chunked content text
- `vector`: number[] - Embedding vector (from Gemini embedding-001)
- `metadata`: object - Additional metadata including:
  - `source_url`: string - URL of the source
  - `module`: string - Module identifier
  - `title`: string - Chapter title
  - `section_heading`: string - Heading of the section (if applicable)

## Validation Rules

### UserProfile Validation
- `hardware_gpu`, `hardware_jetson`, `hardware_robot` must be boolean
- `python_experience` must be between 0-20
- `linux_proficiency` and `rl_experience` must be between 0-10
- `ros_experience` must be one of the allowed values
- `primary_goal` must be one of 'learning', 'building', 'research'
- `skill_level` must be calculated from other fields (derived property)

### Chapter Validation
- `slug` must follow the format "moduleN-section-name/chapter-name"
- `order` must be a positive number
- `url` must be a valid URL path

### Message Validation
- `role` must be 'user' or 'assistant'
- For assistant messages, `sources` must not be empty when answering from textbook
- `content` must not exceed reasonable length limits

### SourceReference Validation
- `relevance_score` must be between 0 and 1
- `url` must be a valid URL path within the textbook

## Relationships

- User 1:1 UserProfile
- User 0:N Conversations
- Conversation 1:N Messages
- Message 0:N SourceReferences
- Chapter 0:N SourceReferences
- Chapter 1:N TextChunks (in Vector DB)

## State Transitions

### Profile State
- New User: Profile is created when user first accesses personalization features
- Profile Completed: When all required profile fields are set
- Profile Updated: When user modifies their profile information

### Conversation State
- Active: When conversation has been recently updated (within 24h)
- Archived: After 30+ days of inactivity (for cleanup purposes)

## Indexes for Performance

### Postgres Database
- User.email (unique, for quick lookups)
- UserProfile.userId (unique, foreign key index)
- Conversation.userId (for user's conversations lookup)
- Message.conversationId (for conversation messages lookup)

### Qdrant Vector Index
- Vector index on the embedding vector (for similarity search)
- Payload index on chapter_id (for filtering by chapter)
- Payload index on source_url (for citation purposes)