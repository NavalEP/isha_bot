// app/api/login/send-otp/route.ts

import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { phone_number } = await req.json();

    if (!phone_number || typeof phone_number !== "string") {
      return NextResponse.json(
        { error: "Phone number is required" },
        { status: 400 }
      );
    }

    console.log("➡️ API call: /api/login/send-otp");
    console.log("📞 Phone number:", phone_number);

    const backendUrl = "http://34.131.33.60/api/v1/agent/login/send-otp/";

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ phone_number }),
    });

    console.log("⬅️ Backend response status:", response.status);

    let data;
    try {
      data = await response.json();
      console.log("✅ Parsed backend response:", data);
    } catch (error) {
      const raw = await response.text();
      console.error("❌ JSON parse failed. Raw response:", raw.substring(0, 500));
      return NextResponse.json(
        { error: "Invalid response from backend" },
        { status: 502 }
      );
    }

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || "Failed to send OTP" },
        { status: response.status }
      );
    }

    return NextResponse.json(data); // Expected format: { message: "OTP sent successfully" }

  } catch (error) {
    console.error("🚨 Exception in /send-otp:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
