import { NextResponse } from 'next/server';

// API base URL from environment variable with fallback
const API_BASE_URL = process.env.CAREPAY_API_URL || 'http://localhost:8000';

// Create a new session
export async function POST() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/agent/session/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json({ 
        status: 'error', 
        message: errorData.message || 'Failed to create session' 
      }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error creating session:', error);
    return NextResponse.json({ 
      status: 'error', 
      message: error instanceof Error ? error.message : 'An unknown error occurred' 
    }, { status: 500 });
  }
}

// Get session status
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get('session_id');

    if (!sessionId) {
      return NextResponse.json({ 
        status: 'error', 
        message: 'Session ID is required' 
      }, { status: 400 });
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/agent/session/${sessionId}/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json({ 
        status: 'error', 
        message: errorData.message || 'Failed to get session status' 
      }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error getting session status:', error);
    return NextResponse.json({ 
      status: 'error', 
      message: error instanceof Error ? error.message : 'An unknown error occurred' 
    }, { status: 500 });
  }
} 