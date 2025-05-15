import { NextResponse } from 'next/server';

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
    const requestBody: Record<string, any> = { phone_number, otp };
    if (doctorId) requestBody.doctorId = doctorId;
    if (doctorName) requestBody.doctorName = doctorName;

    const backendUrl = "http://34.131.33.60/api/v1/agent/login/verify-otp/";

    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    console.log("Backend response status:", response.status);

    let data: any;
    try {
      data = await response.json();
      console.log("Backend response data:", data);
    } catch {
      const text = await response.text();
      console.error("Failed to parse JSON response. Raw:", text);
      return NextResponse.json({ error: 'Invalid response from server' }, { status: 500 });
    }

    if (!response.ok) {
      return NextResponse.json({ error: data.error || 'Invalid OTP' }, { status: response.status });
    }

    // ✅ Set cookies via response headers so middleware can see them
    const cookieOptions = `Path=/; HttpOnly; Max-Age=604800; SameSite=Strict`;

    const res = NextResponse.json({
      message: "OTP verified successfully",
    });

    res.headers.append('Set-Cookie', `auth_token=${data.token}; ${cookieOptions}`);
    res.headers.append('Set-Cookie', `phone_number=${phone_number}; ${cookieOptions}`);
    if (doctorId) res.headers.append('Set-Cookie', `doctor_id=${doctorId}; ${cookieOptions}`);
    if (doctorName) res.headers.append('Set-Cookie', `doctor_name=${doctorName}; ${cookieOptions}`);

    console.log("✅ Cookies set in response headers");

    return res;

  } catch (error) {
    console.error('Error in verify-otp route:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}
