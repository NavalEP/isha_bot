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
  logout: () => Promise<void>;
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

  const login = (token: string, phoneNumber: string, doctorId: string | null = null, doctorName: string | null = null) => {
    // Still use localStorage for client-side checks (cookies are handled by API route)
    localStorage.setItem("auth_token", token);
    localStorage.setItem("phone_number", phoneNumber);
    
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

  const logout = async (): Promise<void> => {
    // First clear localStorage
    localStorage.removeItem("auth_token");
    localStorage.removeItem("phone_number");
    localStorage.removeItem("doctor_id");
    localStorage.removeItem("doctor_name");
    
    // Also clear any session-related data
    localStorage.removeItem("carepay_sessions");
    
    // Clear cookies via API
    try {
      const response = await fetch('/api/logout', { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        console.error('Logout API returned error:', await response.text());
      }
    } catch (error) {
      console.error('Error calling logout API:', error);
    }
    
    // Update state
    setToken(null);
    setPhoneNumber(null);
    setDoctorId(null);
    setDoctorName(null);
    setIsAuthenticated(false);
    
    // Force a hard navigation to ensure all state is cleared
    window.location.href = "/login";
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