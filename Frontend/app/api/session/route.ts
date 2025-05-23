import { NextResponse } from 'next/server';

// API base URL from environment variable with fallback
// const API_BASE_URL ='';

// Create a new session
export async function POST() {
  try {
    const response = await fetch(`http://34.131.33.60/api/v1/agent/session/`, {
      method: 'POST',
      headers: {
        // 'Authorization': `Bearer ${yourJWTToken}`,
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

