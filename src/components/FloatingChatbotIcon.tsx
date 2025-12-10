import React, { useState } from 'react';
import { ChatBubbleLeftRightIcon, XMarkIcon } from '@heroicons/react/24/outline';
import Chatbot from './Chatbot';
import styles from './FloatingChatbotIcon.module.css';

export default function FloatingChatbotIcon(): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Floating Icon Button */}
      <button
        className={styles.floatingButton}
        onClick={toggleChatbot}
        aria-label="Toggle AI Chatbot"
        title="Ask AI about the content"
      >
        {isOpen ? (
          <XMarkIcon className={styles.icon} />
        ) : (
          <ChatBubbleLeftRightIcon className={styles.icon} />
        )}
      </button>

      {/* Chatbot Panel */}
      {isOpen && (
        <div className={styles.chatbotContainer}>
          <Chatbot onClose={() => setIsOpen(false)} />
        </div>
      )}
    </>
  );
}
