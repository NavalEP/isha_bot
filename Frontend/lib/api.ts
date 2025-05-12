/**
 * Utility functions for making authenticated API requests
 */

/**
 * Get the authentication token from localStorage
 */
export const getAuthToken = (): string | null => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('auth_token');
  }
  return null;
};

/**
 * Make an authenticated API request
 */
export const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
  const token = getAuthToken();
  
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...options.headers,
  };
  
  return fetch(url, {
    ...options,
    headers,
  });
};

/**
 * Make a GET request to the backend API
 */
export const apiGet = async (endpoint: string) => {
  const response = await fetchWithAuth(`${process.env.NEXT_PUBLIC_API_URL}/${endpoint}`);
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
};

/**
 * Make a POST request to the backend API
 */
export const apiPost = async (endpoint: string, data: any) => {
  const response = await fetchWithAuth(`${process.env.NEXT_PUBLIC_API_URL}/${endpoint}`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}; 