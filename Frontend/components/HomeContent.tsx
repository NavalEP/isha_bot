"use client";

import Image from "next/image";
import ChatInterface from "@/components/ChatInterface";
import AppHeader from "@/components/AppHeader";
import ChatHistory from "@/components/ChatHistory";
import { useState, useEffect, useRef } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useRouter } from "next/navigation";
import { ChatSession } from "@/types/chat";

// Define welcome message that will be used for new chats
const WELCOME_MESSAGE = "Hello! I'm here to assist you with your healthcare loan application. Let's get started.\n\nFirst, could you please provide me with the following details?\n1. Your full name\n2. Your phone number\n3. The cost of the treatment you are seeking\n4. Your monthly income";

export default function HomeContent() {
  const [showHistory, setShowHistory] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [chatCount, setChatCount] = useState(0);
  const inactivityTimerRef = useRef<NodeJS.Timeout | null>(null);
  const { logout } = useAuth();
  const router = useRouter();

  // Load chat sessions from localStorage on component mount
  useEffect(() => {
    const storedSessions = localStorage.getItem('chat_sessions');
    if (storedSessions) {
      try {
        const parsedSessions = JSON.parse(storedSessions);
        // Convert string dates back to Date objects
        const sessions = parsedSessions.map((session: any) => ({
          ...session,
          timestamp: new Date(session.timestamp)
        }));
        setChatSessions(sessions);
      } catch (error) {
        console.error('Error parsing chat sessions:', error);
      }
    }
    
    // Initialize chat count from localStorage
    const storedChatCount = localStorage.getItem('chat_count');
    if (storedChatCount) {
      setChatCount(parseInt(storedChatCount, 10));
    }
  }, []);

  // Auto-logout after 10 new chats
  useEffect(() => {
    if (chatCount >= 10) {
      // Reset chat count
      setChatCount(0);
      localStorage.setItem('chat_count', '0');
      
      // Logout
      logout();
      router.push('/login');
    } else {
      // Save chat count to localStorage
      localStorage.setItem('chat_count', chatCount.toString());
    }
  }, [chatCount, logout, router]);

  // Create a new chat session
  const handleNewChat = () => {
    // Generate a new session ID
    const newSessionId = `session_${Date.now()}`;
    
    // Create a new session
    const newSession: ChatSession = {
      id: newSessionId,
      timestamp: new Date(),
      preview: WELCOME_MESSAGE.split('\n')[0] // Use first line as preview
    };
    
    // Add to sessions list
    const updatedSessions = [newSession, ...chatSessions];
    setChatSessions(updatedSessions);
    
    // Save to localStorage
    localStorage.setItem('chat_sessions', JSON.stringify(updatedSessions));
    
    // Set as current session
    setCurrentSessionId(newSessionId);
    
    // Increment chat count
    setChatCount(prevCount => prevCount + 1);
    
    // Close history if open
    setShowHistory(false);
  };

  // Handle selecting a session from history
  const handleSelectSession = (sessionId: string) => {
    setCurrentSessionId(sessionId);
    setShowHistory(false);
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
      {/* Mobile phone frame */}
      <div className="relative w-full max-w-md h-[700px] bg-white rounded-[40px] shadow-xl overflow-hidden border-8 border-gray-800">
        {/* Phone notch */}
        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1/3 h-6 bg-gray-800 rounded-b-xl z-10"></div>
        
        {/* App content */}
        <div className="h-full flex flex-col mt-6">
          <AppHeader 
            onNewChat={handleNewChat}
            onViewHistory={() => setShowHistory(true)}
          />
          
          <div className="flex-1 overflow-hidden relative">
            {/* Chat History overlay */}
            {showHistory && (
              <ChatHistory 
                sessions={chatSessions}
                onSelectSession={handleSelectSession}
                onClose={() => setShowHistory(false)}
              />
            )}
            
            {/* Chat Interface */}
            <ChatInterface 
              key={currentSessionId || 'default'}
              initialMessage={WELCOME_MESSAGE}
              sessionId={currentSessionId}
              onNewChat={handleNewChat}
            />
          </div>
        </div>
        
        {/* Phone home indicator */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 w-1/3 h-1 bg-gray-800 rounded-full"></div>
      </div>
    </main>
  );
} 