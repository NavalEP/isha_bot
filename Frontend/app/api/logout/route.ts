import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST() {
  const cookieStore = cookies();
  
  // Get doctor information before clearing cookies
  const doctorId = cookieStore.get('doctor_id')?.value || null;
  const doctorName = cookieStore.get('doctor_name')?.value || null;
  
  // Clear authentication cookies
  cookieStore.delete('auth_token');
  cookieStore.delete('phone_number');
  cookieStore.delete('doctor_id');
  cookieStore.delete('doctor_name');
  
  return NextResponse.json({
    success: true,
    doctorId,
    doctorName
  });
} 