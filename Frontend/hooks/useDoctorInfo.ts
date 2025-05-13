import { useState, useEffect } from 'react';
import { useAuth } from './useAuth';

interface DoctorInfo {
  doctorId: string | null;
  doctorName: string | null;
  isLoading: boolean;
  error: string | null;
  hasDoctorInfo: boolean;
}

export function useDoctorInfo(): DoctorInfo {
  const { isAuthenticated } = useAuth();
  const [doctorInfo, setDoctorInfo] = useState<DoctorInfo>({
    doctorId: null,
    doctorName: null,
    isLoading: true,
    error: null,
    hasDoctorInfo: false
  });

  useEffect(() => {
    if (!isAuthenticated) {
      setDoctorInfo(prev => ({
        ...prev,
        isLoading: false,
        hasDoctorInfo: false
      }));
      return;
    }

    const fetchDoctorInfo = async () => {
      try {
        const response = await fetch('/api/session/doctor');
        
        if (!response.ok) {
          throw new Error('Failed to fetch doctor information');
        }
        
        const data = await response.json();
        
        // Only set doctor information if it exists
        setDoctorInfo({
          doctorId: data.doctorId || null,
          doctorName: data.doctorName || null,
          isLoading: false,
          error: null,
          hasDoctorInfo: data.hasDoctorInfo
        });
      } catch (error) {
        console.error('Error fetching doctor info:', error);
        setDoctorInfo(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          hasDoctorInfo: false
        }));
      }
    };

    fetchDoctorInfo();
  }, [isAuthenticated]);

  return doctorInfo;
} 