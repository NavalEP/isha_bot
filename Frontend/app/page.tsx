import { Metadata } from "next";
import dynamic from 'next/dynamic';

// Import the HomeContent component with dynamic import to prevent SSR issues
const HomeContent = dynamic(() => import('../components/HomeContent'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center min-h-screen">Loading...</div>
});

export const metadata: Metadata = {
  title: "CarePay Loan Assistant",
  description: "AI-powered healthcare financing assistant",
};

export default function Home() {
  return <HomeContent />;
}