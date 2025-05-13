import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST(req: Request) {
  try {
    const { phone_number, otp, doctorId, doctorName } = await req.json();

    if (!phone_number || !otp) {
      return NextResponse.json(
        { error: 'Phone number and OTP are required' },
        { status: 400 }
      );
    }

    console.log("Verifying OTP for phone:", phone_number);
    if (doctorId) console.log("With doctorId:", doctorId);
    if (doctorName) console.log("With doctorName:", doctorName);
    
    // Use a hardcoded URL for now to troubleshoot
    const backendUrl = "http://localhost:8000";
    
    // Only include doctor information in the request if it exists
    const requestBody: any = { phone_number, otp };
    if (doctorId) requestBody.doctorId = doctorId;
    if (doctorName) requestBody.doctorName = doctorName;
    
    // Forward the request to the backend API
    const response = await fetch(`${backendUrl}/api/v1/agent/login/verify-otp/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
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

    // Only set doctor cookies if the information exists
    if (doctorId) {
      console.log("Setting doctor_id cookie with value:", doctorId);
      cookieStore.set('doctor_id', doctorId, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 7 * 24 * 60 * 60, // 7 days
        path: '/',
        sameSite: 'strict',
      });
    }

    if (doctorName) {
      console.log("Setting doctor_name cookie with value:", doctorName);
      cookieStore.set('doctor_name', doctorName, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 7 * 24 * 60 * 60, // 7 days
        path: '/',
        sameSite: 'strict',
      });
    }

    // Return only the doctor information that exists in the response
    const responseBody: any = {
      message: "OTP verified successfully",
      token: data.token,
      phone_number: phone_number
    };

    if (doctorId) responseBody.doctorId = doctorId;
    if (doctorName) responseBody.doctorName = doctorName;

    return NextResponse.json(responseBody);
  } catch (error) {
    console.error('Error verifying OTP:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
} 