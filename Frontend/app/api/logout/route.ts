import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST() {
  const cookieStore = cookies();
  
  // Clear authentication cookies with proper options
  // Setting maxAge to 0 and expires to a past date ensures immediate expiration
  const cookieOptions = {
    maxAge: 0,
    expires: new Date(0),
    path: '/',
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    sameSite: 'strict' as const
  };
  
  cookieStore.delete('auth_token');
  cookieStore.delete('phone_number');
  cookieStore.delete('doctor_id');
  cookieStore.delete('doctor_name');
  
  return NextResponse.json(
    { success: true, message: "Logged out successfully" },
    { status: 200 }
  );
} 