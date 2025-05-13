'use client';

import { useDoctorInfo } from "@/hooks/useDoctorInfo";

export function DoctorInfoDisplay() {
  const { doctorId, doctorName, isLoading, hasDoctorInfo } = useDoctorInfo();

  if (isLoading) {
    return <div className="text-sm text-gray-500">Loading doctor information...</div>;
  }

  // Don't render anything if there's no doctor information
  if (!hasDoctorInfo || !doctorId || !doctorName) {
    return null;
  }

  return (
    <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
      <h3 className="text-sm font-medium text-blue-800">Doctor Information</h3>
      <p className="text-sm text-blue-700">
        Dr. {doctorName.replace(/_/g, ' ')}
      </p>
      <p className="text-xs text-blue-600 opacity-75">ID: {doctorId}</p>
    </div>
  );
} 