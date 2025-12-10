import { useState, useCallback } from 'react';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const queryGemini = useCallback(async (prompt) => {
    setLoading(true);
    setError(null);
    
    try {
      // In a real implementation, this would call your backend API
      // which interfaces with Gemini
      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        response: data.response,
        sources: data.sources || []
      };
    } catch (err) {
      setError(err.message);
      console.error('Gemini API error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const queryRAG = useCallback(async (query, context = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query,
          context,
          top_k: 5  // Get top 5 relevant results
        }),
      });

      if (!response.ok) {
        throw new Error(`RAG query failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      console.error('RAG query error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    queryGemini,
    queryRAG,
    loading,
    error
  };
};