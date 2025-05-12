import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST(req: Request) {
  try {
    const { phone_number, otp } = await req.json();

    if (!phone_number || !otp) {
      return NextResponse.json(
        { error: 'Phone number and OTP are required' },
        { status: 400 }
      );
    }

    console.log("Verifying OTP for phone:", phone_number);
    
    // Use a hardcoded URL for now to troubleshoot
    const backendUrl = "http://localhost:8000";
    
    // Forward the request to the backend API
    const response = await fetch(`${backendUrl}/api/v1/agent/login/verify-otp/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ phone_number, otp }),
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
        { error: data.error || 'Invalid OTP' },
        { status: response.status }
      );
    }

    // Set cookies for authentication
    const cookieStore = cookies();
    
    console.log("Setting auth_token cookie with value:", data.token);
    
    cookieStore.set('auth_token', data.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      maxAge: 7 * 24 * 60 * 60, // 7 days
      path: '/',
      sameSite: 'strict',
    });
    
    console.log("Setting phone_number cookie with value:", phone_number);
    
    cookieStore.set('phone_number', phone_number, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      maxAge: 7 * 24 * 60 * 60, // 7 days
      path: '/',
      sameSite: 'strict',
    });
    
    // Set doctor_id and doctor_name cookies if they exist in the response
    if (data.doctor_id) {
      console.log("Setting doctor_id cookie with value:", data.doctor_id);
      cookieStore.set('doctor_id', data.doctor_id, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 7 * 24 * 60 * 60, // 7 days
        path: '/',
        sameSite: 'strict',
      });
    }
    
    if (data.doctor_name) {
      console.log("Setting doctor_name cookie with value:", data.doctor_name);
      cookieStore.set('doctor_name', data.doctor_name, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 7 * 24 * 60 * 60, // 7 days
        path: '/',
        sameSite: 'strict',
      });
    }

    // Return the token in the response body as well
    return NextResponse.json({
      message: "OTP verified successfully",
      token: data.token,
      phone_number: phone_number,
      doctor_id: data.doctor_id,
      doctor_name: data.doctor_name
    });
  } catch (error) {
    console.error('Error verifying OTP:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
} 