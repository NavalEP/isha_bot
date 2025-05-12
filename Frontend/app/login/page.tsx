"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import Image from "next/image";

export default function LoginPage() {
  const router = useRouter();
  const { toast } = useToast();
  const { login } = useAuth();
  const [phone, setPhone] = useState("");
  const [otp, setOtp] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [otpSent, setOtpSent] = useState(false);

  const sendOtp = async () => {
    if (!phone || phone.length < 10) {
      toast({
        variant: "destructive",
        title: "Invalid phone number",
        description: "Please enter a valid phone number",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch("/api/login/send-otp", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ phone_number: phone }),
      });

      const data = await response.json();

      if (response.ok) {
        toast({
          title: "OTP Sent",
          description: "Please check your phone for the OTP",
        });
        setOtpSent(true);
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: data.error || "Failed to send OTP. Please try again.",
        });
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Something went wrong. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const verifyOtp = async () => {
    if (!otp) {
      toast({
        variant: "destructive",
        title: "OTP Required",
        description: "Please enter the OTP sent to your phone",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch("/api/login/verify-otp", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ phone_number: phone, otp }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store the token and use the auth hook with doctor details
        login(data.token, phone, data.doctor_id, data.doctor_name);
        
        toast({
          title: "Login Successful",
          description: "Redirecting to chat...",
        });
        
        // Add a delay before redirecting to ensure state is updated
        setTimeout(() => {
          // Force a hard navigation to ensure middleware picks up the new cookies
          window.location.href = "/";
        }, 1000);
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: data.error || "Invalid OTP. Please try again.",
        });
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Something went wrong. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
      {/* Mobile phone frame */}
      <div className="relative w-full max-w-md h-[700px] bg-white rounded-[40px] shadow-xl overflow-hidden border-8 border-gray-800">
        {/* Phone notch */}
        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1/3 h-6 bg-gray-800 rounded-b-xl z-10"></div>
        
        {/* Login content */}
        <div className="flex flex-col items-center justify-center h-full px-8 pt-12 pb-8">
          <div className="w-full max-w-xs">
            {/* Logo and title */}
            <div className="text-center mb-8">
              <div className="flex justify-center mb-2">
                <Image
                  src="/logo.svg"
                  alt="CarePay Logo"
                  width={48}
                  height={48}
                  className="h-12 w-auto"
                />
              </div>
              <h1 className="text-2xl font-bold text-blue-600 mb-1">
                CarePay Assistant
              </h1>
              <p className="text-sm text-gray-500">
                Login to access your healthcare assistant
              </p>
            </div>
            
            <div className="space-y-6">
              {/* Phone input */}
              <div className="space-y-2">
                <label htmlFor="phone" className="text-sm font-medium">
                  Phone Number
                </label>
                <Input
                  id="phone"
                  placeholder="Enter your phone number"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  disabled={otpSent}
                  className="w-full"
                />
              </div>

              {!otpSent ? (
                <Button 
                  onClick={sendOtp} 
                  className="w-full" 
                  disabled={isLoading}
                >
                  {isLoading ? "Sending..." : "Send OTP"}
                </Button>
              ) : (
                <>
                  <div className="space-y-2">
                    <label htmlFor="otp" className="text-sm font-medium">
                      One-Time Password
                    </label>
                    <Input
                      id="otp"
                      placeholder="Enter OTP"
                      value={otp}
                      onChange={(e) => setOtp(e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <div className="flex flex-col space-y-3">
                    <Button 
                      onClick={verifyOtp} 
                      className="w-full" 
                      disabled={isLoading}
                    >
                      {isLoading ? "Verifying..." : "Verify OTP"}
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => setOtpSent(false)} 
                      className="w-full"
                      disabled={isLoading}
                    >
                      Change Phone Number
                    </Button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
        
        {/* Phone home indicator */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 w-1/3 h-1 bg-gray-800 rounded-full"></div>
      </div>
    </main>
  );
} 