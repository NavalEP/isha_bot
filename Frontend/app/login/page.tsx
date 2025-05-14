"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import Image from "next/image";

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const { login } = useAuth();
  const [phone, setPhone] = useState("");
  const [otp, setOtp] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  
  // Get doctorId and doctor_name from URL parameters
  const doctorId = searchParams.get("doctorId");
  const doctorName = searchParams.get("doctor_name");

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
        body: JSON.stringify({ 
          phone_number: phone, 
          otp,
          doctorId,
          doctorName
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store the token and use the auth hook with doctor information
        login(data.token, phone, doctorId, doctorName);
        
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
        
        {/* App content */}
        <div className="h-full flex flex-col mt-6">
          {/* App header */}
          <header className="bg-white border-b border-border py-3 px-4 sticky top-0 z-10 shadow-sm">
            <div className="flex items-center">
              <div className="flex items-center gap-2">
                <Image
                  src="/logo.svg"
                  alt="CarePay Logo"
                  width={24}
                  height={24}
                  className="h-6 w-auto"
                />
                <h1 className="text-lg font-semibold text-blue-600">
                  CarePay Assistant
                </h1>
              </div>
            </div>
          </header>
          
          <div className="flex-1 flex items-center justify-center p-4">
            <Card className="w-full shadow-none border-none">
              <CardHeader className="space-y-1">
                <CardTitle className="text-2xl font-bold text-center">Login</CardTitle>
                <CardDescription className="text-center">
                  {otpSent ? "Enter the OTP sent to your phone" : "Enter your phone number to receive an OTP"}
                </CardDescription>
                {doctorId && doctorName && (
                  <CardDescription className="text-center text-sm text-green-600">
                    Logging in with Dr. {doctorName.replace(/_/g, ' ')}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Input
                      id="phone"
                      placeholder="Phone Number"
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
                        <Input
                          id="otp"
                          placeholder="Enter OTP"
                          value={otp}
                          onChange={(e) => setOtp(e.target.value)}
                          className="w-full"
                        />
                      </div>
                      <div className="flex flex-col space-y-2">
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
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* Phone home indicator */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 w-1/3 h-1 bg-gray-800 rounded-full"></div>
      </div>
    </main>
  );
} 