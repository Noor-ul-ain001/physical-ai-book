import React from 'react';
import ChatBot from '@site/src/components/Chatbot';

export default function Root({ children }) {
  return (
    <>
      {children}
      <ChatBot />
    </>
  );
}
