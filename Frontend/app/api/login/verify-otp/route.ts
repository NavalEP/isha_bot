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

    console.log("Hit /api/login/verify-otp route");
    console.log("Verifying OTP for:", phone_number);
    if (doctorId) console.log("doctorId:", doctorId);
    if (doctorName) console.log("doctorName:", doctorName);

    const requestBody: Record<string, any> = { phone_number, otp };
    if (doctorId) requestBody.doctorId = doctorId;
    if (doctorName) requestBody.doctorName = doctorName;

    const backendUrl = "http://34.131.33.60/api/v1/agent/login/verify-otp/";

    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    console.log("Backend response status:", response.status);

    let data: any;
    try {
      data = await response.json();
      console.log("Backend response data:", data);
    } catch (jsonError) {
      console.error("Failed to parse JSON response");
      const text = await response.text();
      console.log("Raw response:", text);
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

    const cookieStore = cookies();
    const cookieOptions = {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      maxAge: 7 * 24 * 60 * 60,
      path: '/',
      sameSite: 'strict' as const,
    };

    console.log("Setting cookies...");

    cookieStore.set('auth_token', data.token, cookieOptions);
    cookieStore.set('phone_number', phone_number, cookieOptions);

    if (doctorId) cookieStore.set('doctor_id', doctorId, cookieOptions);
    if (doctorName) cookieStore.set('doctor_name', doctorName, cookieOptions);

    return NextResponse.json({
      message: "OTP verified successfully",
      token: data.token,
      phone_number,
      ...(doctorId && { doctorId }),
      ...(doctorName && { doctorName }),
    });

  } catch (error) {
    console.error('Error in verify-otp route:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
}
