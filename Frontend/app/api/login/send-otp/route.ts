import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { phone_number } = await req.json();

    if (!phone_number) {
      return NextResponse.json(
        { error: 'Phone number is required' },
        { status: 400 }
      );
    }

    console.log("Sending OTP request to backend for phone:", phone_number);
    
    // Use a hardcoded URL for now to troubleshoot
    const API_BASE_URL = process.env.CAREPAY_API_URL || 'http://localhost:8000';  
    
    // Forward the request to the backend API
    const response = await fetch(`${API_BASE_URL}api/v1/agent/login/send-otp/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ phone_number }),
    });

    console.log("Backend response status:", response.status);
    
    let data;
    try {
      data = await response.json();
      console.log("Backend response data:", data);
    } catch (error) {
      console.error("Failed to parse JSON response:", error);
      const textResponse = await response.text();
      console.log("Raw response:", textResponse.substring(0, 500));
      return NextResponse.json(
        { error: 'Invalid response from server' },
        { status: 500 }
      );
    }

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || 'Failed to send OTP' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error sending OTP:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
} 