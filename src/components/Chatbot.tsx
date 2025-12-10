import React, { useState, useEffect, useRef } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

const ChatBot = () => {
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

  const location = typeof window !== 'undefined' ? window.location : { pathname: '/' };

  useEffect(() => {
    if (typeof document !== 'undefined') {
      const htmlElement = document.documentElement;
      const mode = htmlElement.getAttribute('data-theme') || 'light';
      setColorMode(mode);

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

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

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
      const currentPageContext = `The user is currently viewing: ${location.pathname}. `;

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

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
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
            background: colorMode === 'dark' ? '#141E30' : '#F5F7FF',
            color: colorMode === 'dark' ? '#F5F7FF' : '#141E30',
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

      {isOpen && (
        <div
          className={`chatbot-window ${colorMode}`}
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '400px',
            height: '600px',
            backgroundColor: colorMode === 'dark' ? '#141E30' : '#F5F7FF',
            border: `1px solid ${colorMode === 'dark' ? '#0d1524' : '#d4d8f0'}`,
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
              background: colorMode === 'dark' ? '#141E30' : '#F5F7FF',
              color: colorMode === 'dark' ? '#F5F7FF' : '#141E30',
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
                onClick={() => setMessages([])}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  border: 'none',
                  borderRadius: '4px',
                  color: colorMode === 'dark' ? '#F5F7FF' : '#141E30',
                  cursor: 'pointer',
                  padding: '2px 6px',
                  fontSize: '12px'
                }}
              >
                Clear
              </button>
              <button
                onClick={closeChat}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  border: 'none',
                  borderRadius: '50%',
                  color: colorMode === 'dark' ? '#F5F7FF' : '#141E30',
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

          {/* Messages */}
          <div
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              backgroundColor: colorMode === 'dark' ? '#141E30' : '#F5F7FF'
            }}
          >
            {messages.length === 0 ? (
              <div
                style={{
                  textAlign: 'center',
                  color: colorMode === 'dark' ? '#aab7d1' : '#444',
                  fontStyle: 'italic',
                  marginTop: 'auto',
                  marginBottom: 'auto',
                  padding: '20px'
                }}
              >
                <h3>Hello! I'm your Physical AI & Humanoid Robotics assistant.</h3>
                <p>Ask me about:</p>
                <ul style={{ textAlign: 'left', marginLeft: '20px' }}>
                  <li>ROS2 fundamentals</li>
                  <li>Digital twins</li>
                  <li>Isaac Sim</li>
                  <li>Reinforcement learning</li>
                  <li>VLA models</li>
                </ul>
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
                      backgroundColor:
                        message.sender === 'user'
                          ? (colorMode === 'dark' ? '#1f2942' : '#dce3ff')
                          : (colorMode === 'dark' ? '#141E30' : '#ffffff'),
                      border: `1px solid ${colorMode === 'dark' ? '#1b263b' : '#d4d8f0'}`,
                      color: colorMode === 'dark' ? '#F5F7FF' : '#000',
                      wordBreak: 'break-word'
                    }}
                  >
                    <div>{message.text}</div>

                    {message.sources && message.sources.length > 0 && (
                      <div style={{ fontSize: '0.8em', marginTop: '6px' }}>
                        <strong style={{ color: colorMode === 'dark' ? '#F5F7FF' : '#141E30' }}>
                          Sources:
                        </strong>
                        <ul style={{ paddingLeft: '15px' }}>
                          {message.sources.slice(0, 2).map((source, idx) => (
                            <li key={idx}>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  color: colorMode === 'dark' ? '#F5F7FF' : '#141E30',
                                  textDecoration: 'underline'
                                }}
                              >
                                {source.title || 'Related content'}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                <div
                  style={{
                    maxWidth: '85%',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: colorMode === 'dark' ? '#141E30' : '#ffffff',
                    border: `1px solid ${colorMode === 'dark' ? '#1b263b' : '#d4d8f0'}`,
                    color: colorMode === 'dark' ? '#F5F7FF' : '#000'
                  }}
                >
                  <div>🤖 Thinking...</div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div
            style={{
              padding: '12px',
              borderTop: `1px solid ${colorMode === 'dark' ? '#1b263b' : '#d4d8f0'}`,
              backgroundColor: colorMode === 'dark' ? '#141E30' : '#F5F7FF'
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
                  border: `1px solid ${colorMode === 'dark' ? '#1b263b' : '#d4d8f0'}`,
                  backgroundColor: colorMode === 'dark' ? '#1f2942' : '#ffffff',
                  color: colorMode === 'dark' ? '#F5F7FF' : '#000',
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
                  backgroundColor:
                    inputValue.trim() && !isLoading
                      ? (colorMode === 'dark' ? '#F5F7FF' : '#141E30')
                      : (colorMode === 'dark' ? '#1f2942' : '#d4d8f0'),
                  color:
                    inputValue.trim() && !isLoading
                      ? (colorMode === 'dark' ? '#141E30' : '#F5F7FF')
                      : (colorMode === 'dark' ? '#9aa5c3' : '#666'),
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
