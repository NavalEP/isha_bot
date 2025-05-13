'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { useRouter } from "next/navigation";

interface AuthContextType {
  isAuthenticated: boolean;
  phoneNumber: string | null;
  token: string | null;
  doctorId: string | null;
  doctorName: string | null;
  login: (token: string, phoneNumber: string, doctorId?: string | null, doctorName?: string | null) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [phoneNumber, setPhoneNumber] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [doctorId, setDoctorId] = useState<string | null>(null);
  const [doctorName, setDoctorName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // Check if user is authenticated on initial load (client-side only)
    if (typeof window !== 'undefined') {
      // For backward compatibility, check localStorage
      const storedToken = localStorage.getItem("auth_token");
      const storedPhoneNumber = localStorage.getItem("phone_number");
      const storedDoctorId = localStorage.getItem("doctor_id");
      const storedDoctorName = localStorage.getItem("doctor_name");

      if (storedToken && storedPhoneNumber) {
        setToken(storedToken);
        setPhoneNumber(storedPhoneNumber);
        setDoctorId(storedDoctorId);
        setDoctorName(storedDoctorName);
        setIsAuthenticated(true);
      }
      
      setLoading(false);
    }
  }, []);

  const login = (token: string, phoneNumber: string, doctorId?: string | null, doctorName?: string | null) => {
    // Still use localStorage for client-side checks (cookies are handled by API route)
    localStorage.setItem("auth_token", token);
    localStorage.setItem("phone_number", phoneNumber);
    
    // Only store doctor information if it exists
    if (doctorId) {
      localStorage.setItem("doctor_id", doctorId);
      setDoctorId(doctorId);
    }
    
    if (doctorName) {
      localStorage.setItem("doctor_name", doctorName);
      setDoctorName(doctorName);
    }
    
    setToken(token);
    setPhoneNumber(phoneNumber);
    setIsAuthenticated(true);
  };

  const logout = async () => {
    // Clear localStorage
    localStorage.removeItem("auth_token");
    localStorage.removeItem("phone_number");
    
    // Store doctor information before clearing it
    const currentDoctorId = doctorId;
    const currentDoctorName = doctorName;
    
    localStorage.removeItem("doctor_id");
    localStorage.removeItem("doctor_name");
    
    // Clear cookies via API and get doctor information
    let logoutDoctorId = currentDoctorId;
    let logoutDoctorName = currentDoctorName;
    
    try {
      const response = await fetch('/api/logout', { method: 'POST' });
      const data = await response.json();
      
      // Use doctor information from API response if available
      if (data.doctorId) logoutDoctorId = data.doctorId;
      if (data.doctorName) logoutDoctorName = data.doctorName;
    } catch (error) {
      console.error('Error logging out:', error);
    }
    
    setToken(null);
    setPhoneNumber(null);
    setDoctorId(null);
    setDoctorName(null);
    setIsAuthenticated(false);
    
    // Redirect to login page with doctor credentials if available
    if (logoutDoctorId && logoutDoctorName) {
      const loginUrl = `/login?doctorId=${encodeURIComponent(logoutDoctorId)}&doctor_name=${encodeURIComponent(logoutDoctorName)}`;
      router.push(loginUrl);
    } else {
      router.push("/login");
    }
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, phoneNumber, token, doctorId, doctorName, login, logout }}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
} 