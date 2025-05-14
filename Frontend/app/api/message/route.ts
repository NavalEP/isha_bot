import { NextResponse } from 'next/server';

// API base URL from environment variable with fallback
const API_BASE_URL = process.env.CAREPAY_API_URL || 'http://34.131.33.60/api/';  

// Send a message
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message, session_id } = body;

    // Validate input
    if (!message) {
      return NextResponse.json({ 
        status: 'error', 
        message: 'Message is required' 
      }, { status: 400 });
    }

    if (!session_id) {
      return NextResponse.json({ 
        status: 'error', 
        message: 'Session ID is required' 
      }, { status: 400 });
    }

    const response = await fetch(`${API_BASE_URL}v1/agent/message/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json({ 
        status: 'error', 
        message: errorData.message || 'Failed to send message' 
      }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error sending message:', error);
    return NextResponse.json({ 
      status: 'error', 
      message: error instanceof Error ? error.message : 'An unknown error occurred' 
    }, { status: 500 });
  }
} 