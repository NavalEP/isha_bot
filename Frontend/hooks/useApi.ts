import { useState } from 'react';
import { ChatResponse } from '@/types/chat';
import { useAuth } from './useAuth';

// Direct API URL to Django backend
const API_BASE_URL = process.env.CAREPAY_API_URL || 'http://localhost:8000/api/v1/agent';

interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

export const useApi = () => {
  const { token } = useAuth();

  // Helper function to get headers with auth token
  const getHeaders = () => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    return headers;
  };

  // Create a new chat session
  const createSession = async (): Promise<ApiResponse<{ session_id: string }>> => {
    try {
      const response = await fetch(`${API_BASE_URL}/session/`, {
        method: 'POST',
        headers: getHeaders(),
        // Add CORS headers for direct API calls
        mode: 'cors',
        credentials: 'include',
      });

      if (!response.ok) {
        // Try to parse JSON response if possible
        try {
          const errorData = await response.json();
          return { data: null, error: errorData.message || `Failed to create session (${response.status})`, loading: false };
        } catch (parseError) {
          return { data: null, error: `Server error: ${response.status} ${response.statusText}`, loading: false };
        }
      }

      const data = await response.json();
      return { data, error: null, loading: false };
    } catch (error) {
      console.error("API error:", error);
      return { 
        data: null, 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        loading: false 
      };
    }
  };

  // Send message to the chat API
  const sendChatMessage = async (message: string, sessionId?: string): Promise<ApiResponse<ChatResponse>> => {
    try {
      // If no session ID is provided, create a new session first
      if (!sessionId) {
        const sessionResponse = await createSession();
        if (sessionResponse.error) {
          return { data: null, error: sessionResponse.error, loading: false };
        }
        sessionId = sessionResponse.data?.session_id;
      }

      const response = await fetch(`${API_BASE_URL}/message/`, {
        method: 'POST',
        headers: getHeaders(),
        // Add CORS headers for direct API calls
        mode: 'cors',
        credentials: 'include',
        body: JSON.stringify({
          session_id: sessionId,
          message,
        }),
      });

      if (!response.ok) {
        // Try to parse JSON response if possible
        try {
          const errorData = await response.json();
          return { data: null, error: errorData.message || `Failed to send message (${response.status})`, loading: false };
        } catch (parseError) {
          return { data: null, error: `Server error: ${response.status} ${response.statusText}`, loading: false };
        }
      }

      const data = await response.json();
      return { data, error: null, loading: false };
    } catch (error) {
      console.error("API error:", error);
      return { 
        data: null, 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        loading: false 
      };
    }
  };

  // Get session status
  const getSessionStatus = async (sessionId: string): Promise<ApiResponse<any>> => {
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/`, {
        method: 'GET',
        headers: getHeaders(),
        // Add CORS headers for direct API calls
        mode: 'cors',
        credentials: 'include',
      });

      if (!response.ok) {
        // Try to parse JSON response if possible
        try {
          const errorData = await response.json();
          return { data: null, error: errorData.message || `Failed to get session status (${response.status})`, loading: false };
        } catch (parseError) {
          return { data: null, error: `Server error: ${response.status} ${response.statusText}`, loading: false };
        }
      }

      const data = await response.json();
      return { data, error: null, loading: false };
    } catch (error) {
      console.error("API error:", error);
      return { 
        data: null, 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        loading: false 
      };
    }
  };

  return {
    createSession,
    sendChatMessage,
    getSessionStatus,
  };
};

export default useApi; 