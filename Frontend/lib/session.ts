import { cookies } from 'next/headers';

export interface SessionData {
  isAuthenticated: boolean;
  phoneNumber: string | null;
  token: string | null;
  doctorId: string | null;
  doctorName: string | null;
}

export function getSessionData(): SessionData {
  const cookieStore = cookies();
  const token = cookieStore.get('auth_token')?.value || null;
  const phoneNumber = cookieStore.get('phone_number')?.value || null;
  const doctorId = cookieStore.get('doctor_id')?.value || null;
  const doctorName = cookieStore.get('doctor_name')?.value || null;

  return {
    isAuthenticated: !!token,
    phoneNumber,
    token,
    doctorId,
    doctorName
  };
}

export function getDoctorInfo() {
  const { doctorId, doctorName } = getSessionData();
  const hasDoctorInfo = !!(doctorId && doctorName);
  
  return { 
    doctorId, 
    doctorName, 
    hasDoctorInfo 
  };
} 