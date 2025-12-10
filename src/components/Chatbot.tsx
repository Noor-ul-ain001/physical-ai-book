import React, { useState, useEffect, useRef } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

const ChatBot = () => {
  // Don't render during SSR
  if (!ExecutionEnvironment.canUseDOM) {
    return null;
  }

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [colorMode, setColorMode] = useState('light');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Get current location path
  const location = typeof window !== 'undefined' ? window.location : { pathname: '/' };

  // Detect color mode from document
  useEffect(() => {
    if (typeof document !== 'undefined') {
      const htmlElement = document.documentElement;
      const mode = htmlElement.getAttribute('data-theme') || 'light';
      setColorMode(mode);

      // Watch for theme changes
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.attributeName === 'data-theme') {
            const newMode = htmlElement.getAttribute('data-theme') || 'light';
            setColorMode(newMode);
          }
        });
      });

      observer.observe(htmlElement, {
        attributes: true,
        attributeFilter: ['data-theme']
      });

      return () => observer.disconnect();
    }
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Send message to backend
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get context about current page
      const currentPageContext = `The user is currently viewing: ${location.pathname}. `;

      // Call the actual backend API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentPageContext + inputValue,
          history: messages.slice(-5).map(m => ({
            id: m.id,
            text: m.text,
            sender: m.sender,
            timestamp: m.timestamp.toISOString()
          }))
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Get user context if available
  const getUserContext = () => {
    // Try to get user profile from localStorage or context
    try {
      const profile = localStorage.getItem('humanoidProfile');
      return profile ? JSON.parse(profile) : null;
    } catch {
      return null;
    }
  };

  // Toggle chat window
  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 100);
    }
  };

  // Close chat
  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
      {/* Chat Icon Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="chatbot-icon-btn"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            backgroundColor: '#76b900',
            color: 'white',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            zIndex: 10000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.3s ease'
          }}
          aria-label="Open chatbot"
        >
          🤖
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div
          className={`chatbot-window ${colorMode}`}
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '400px',
            height: '600px',
            backgroundColor: colorMode === 'dark' ? '#2d2d2d' : '#ffffff',
            border: `1px solid ${colorMode === 'dark' ? '#4a4a4a' : '#ddd'}`,
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 10000,
            boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
          }}
        >
          {/* Header */}
          <div
            style={{
              backgroundColor: '#76b900',
              color: 'white',
              padding: '12px 16px',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span style={{ fontWeight: 'bold', fontSize: '16px' }}>
              Physical AI Assistant
            </span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => {
                  setMessages([]);
                }}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  padding: '2px 6px',
                  fontSize: '12px'
                }}
                title="Clear chat"
              >
                Clear
              </button>
              <button
                onClick={closeChat}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: 'none',
                  borderRadius: '50%',
                  color: 'white',
                  cursor: 'pointer',
                  width: '24px',
                  height: '24px',
                  lineHeight: '20px'
                }}
              >
                ×
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              backgroundColor: colorMode === 'dark' ? '#1e1e1e' : '#fafafa'
            }}
          >
            {messages.length === 0 ? (
              <div
                style={{
                  textAlign: 'center',
                  color: colorMode === 'dark' ? '#aaa' : '#666',
                  fontStyle: 'italic',
                  marginTop: 'auto',
                  marginBottom: 'auto',
                  padding: '20px'
                }}
              >
                <h3>Hello! I'm your Physical AI & Humanoid Robotics assistant.</h3>
                <p>Ask me about:</p>
                <ul style={{ textAlign: 'left', listStyle: 'disc', marginLeft: '20px', marginTop: '10px' }}>
                  <li>ROS2 fundamentals and implementation</li>
                  <li>Digital twin concepts and simulation</li>
                  <li>Isaac Sim and vSLAM techniques</li>
                  <li>Reinforcement learning for humanoid control</li>
                  <li>Vision-language-action models</li>
                  <li>Any content from the textbook chapters</li>
                </ul>
                <p style={{ marginTop: '15px' }}>I have access to the full textbook content and can provide personalized answers based on your expertise level.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  style={{
                    display: 'flex',
                    justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                    width: '100%'
                  }}
                >
                  <div
                    style={{
                      maxWidth: '85%',
                      padding: '10px 14px',
                      borderRadius: '18px',
                      backgroundColor: message.sender === 'user'
                        ? (colorMode === 'dark' ? '#4a4a4a' : '#e3f2fd')
                        : (colorMode === 'dark' ? '#3a3a3a' : '#ffffff'),
                      border: `1px solid ${colorMode === 'dark' ? '#555' : '#ddd'}`,
                      wordWrap: 'break-word',
                      wordBreak: 'break-word'
                    }}
                  >
                    <div
                      style={{
                        marginBottom: message.sources && message.sources.length > 0 ? '8px' : '0'
                      }}
                    >
                      {message.text}
                    </div>

                    {message.sources && message.sources.length > 0 && (
                      <div style={{
                        fontSize: '0.8em',
                        marginTop: '5px',
                        paddingTop: '5px',
                        borderTop: `1px solid ${colorMode === 'dark' ? '#555' : '#eee'}`
                      }}>
                        <strong style={{ color: '#76b900' }}>Sources:</strong>
                        <ul style={{ margin: '5px 0 0 0', paddingLeft: '15px' }}>
                          {message.sources.slice(0, 2).map((source, idx) => (
                            <li key={idx}>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  color: '#76b900',
                                  textDecoration: 'none',
                                  fontSize: '0.9em'
                                }}
                              >
                                {source.title || 'Related content'}
                              </a>
                            </li>
                          ))}
                          {message.sources.length > 2 && (
                            <li style={{ color: '#76b900', fontSize: '0.8em' }}>
                              ... and {message.sources.length - 2} more
                            </li>
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start', width: '100%' }}>
                <div
                  style={{
                    maxWidth: '85%',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: colorMode === 'dark' ? '#3a3a3a' : '#ffffff',
                    border: `1px solid ${colorMode === 'dark' ? '#555' : '#ddd'}`,
                  }}
                >
                  <div>🤖 Thinking...</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            style={{
              padding: '12px',
              borderTop: `1px solid ${colorMode === 'dark' ? '#4a4a4a' : '#eee'}`,
              backgroundColor: colorMode === 'dark' ? '#282828' : '#ffffff'
            }}
          >
            <div style={{ display: 'flex', gap: '8px' }}>
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask about Physical AI & Humanoid Robotics..."
                style={{
                  flex: 1,
                  padding: '10px 12px',
                  borderRadius: '20px',
                  border: `1px solid ${colorMode === 'dark' ? '#4a4a4a' : '#ddd'}`,
                  backgroundColor: colorMode === 'dark' ? '#3a3a3a' : '#ffffff',
                  color: colorMode === 'dark' ? '#fff' : '#000',
                  resize: 'none',
                  minHeight: '40px',
                  maxHeight: '100px',
                  fontSize: '14px'
                }}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                style={{
                  minWidth: '60px',
                  padding: '10px 16px',
                  backgroundColor: inputValue.trim() && !isLoading
                    ? '#76b900'
                    : (colorMode === 'dark' ? '#4a4a4a' : '#e0e0e0'),
                  color: inputValue.trim() && !isLoading ? 'white' : (colorMode === 'dark' ? '#aaa' : '#888'),
                  border: 'none',
                  borderRadius: '20px',
                  cursor: inputValue.trim() && !isLoading ? 'pointer' : 'default',
                  fontWeight: 'bold',
                  fontSize: '14px'
                }}
              >
                {isLoading ? '...' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;