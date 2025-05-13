import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function GET() {
  const cookieStore = cookies();
  const doctorId = cookieStore.get('doctor_id')?.value || null;
  const doctorName = cookieStore.get('doctor_name')?.value || null;
  const hasDoctorInfo = !!(doctorId && doctorName);

  if (!hasDoctorInfo) {
    return NextResponse.json({
      doctorId: null,
      doctorName: null,
      hasDoctorInfo: false,
      message: "No doctor information found in session"
    });
  }

  return NextResponse.json({
    doctorId,
    doctorName,
    hasDoctorInfo: true
  });
} 