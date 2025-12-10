# API Contract: Physical AI & Humanoid Robotics Interactive Textbook

## Chat API

### POST /api/chat

Initiate a conversation with the AI assistant about the textbook content. The assistant provides responses based on the textbook content with source citations.

#### Request Body
```json
{
  "message": "What is the difference between ROS1 and ROS2?",
  "userId": "optional-user-id",
  "conversationId": "optional-conversation-id"
}
```

#### Response
```json
{
  "id": "message-id",
  "response": "Detailed response based on textbook content...",
  "sources": [
    {
      "title": "Module 1: ROS2 Overview",
      "url": "/docs/module1-ros2/01-overview",
      "snippet": "ROS2 was designed to address limitations in ROS1...",
      "relevance_score": 0.87
    }
  ],
  "conversationId": "new-or-existing-conversation-id"
}
```

### POST /api/highlight-chat

Ask about a specific text selection only, using only that context for the response.

#### Request Body
```json
{
  "selectedText": "The robot operating system (ROS) is a flexible framework...",
  "question": "Explain this concept in simpler terms",
  "userId": "optional-user-id",
  "conversationId": "optional-conversation-id"
}
```

#### Response
```json
{
  "id": "message-id",
  "response": "Simplified explanation based only on the selected text...",
  "sources": [
    {
      "title": "Introduction to ROS",
      "url": "/docs/intro/01-foundations",
      "snippet": "The robot operating system (ROS) is a flexible framework...",
      "relevance_score": 0.95
    }
  ],
  "conversationId": "new-or-existing-conversation-id"
}
```

## Content Indexing API

### POST /api/index

Run the content indexing process to update the vector database with all content from MDX files.

#### Request Body (empty)

#### Response
```json
{
  "status": "success",
  "indexedCount": 250,
  "details": "Content indexing completed successfully"
}
```

## User Profile API

### PUT /api/profile

Update the user profile with hardware and experience information.

#### Request Body
```json
{
  "hardware_gpu": true,
  "hardware_jetson": false,
  "hardware_robot": true,
  "python_experience": 5,
  "ros_experience": "ros2",
  "linux_proficiency": 7,
  "rl_experience": 3,
  "primary_goal": "building"
}
```

#### Response
```json
{
  "status": "success",
  "profileId": "profile-id-here",
  "message": "Profile updated successfully"
}
```

### GET /api/profile

Retrieve the current user's profile information.

#### Response
```json
{
  "id": "profile-id-here",
  "userId": "user-id-here",
  "hardware_gpu": true,
  "hardware_jetson": false,
  "hardware_robot": true,
  "python_experience": 5,
  "ros_experience": "ros2",
  "linux_proficiency": 7,
  "rl_experience": 3,
  "primary_goal": "building",
  "skill_level": "intermediate",
  "createdAt": "2023-01-01T00:00:00Z",
  "updatedAt": "2023-01-02T00:00:00Z"
}
```