import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

// This is a mock implementation since we're storing chat history in localStorage
// In a real app, this would fetch data from the backend
export async function GET() {
  const cookieStore = cookies();
  const token = cookieStore.get('auth_token');
  
  if (!token) {
    return NextResponse.json(
      { error: 'Authentication required' },
      { status: 401 }
    );
  }
  
  // In a real implementation, we would fetch chat history from the backend
  // For now, we'll return an empty array since history is managed client-side
  return NextResponse.json({ sessions: [] });
} 