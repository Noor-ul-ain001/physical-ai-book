// index.js - Main entry point for the Physical AI & Humanoid Robotics Interactive Textbook

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './src/App';
import './src/css/custom.css';

// Main application component
function MainApp() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/docs/*" element={<App />} />
        <Route path="/modules/*" element={<App />} />
      </Routes>
    </BrowserRouter>
  );
}

// Render the application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MainApp />
  </React.StrictMode>
);

export default MainApp;