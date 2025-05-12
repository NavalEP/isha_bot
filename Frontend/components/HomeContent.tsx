"use client";

import Image from "next/image";
import ChatInterface from "@/components/ChatInterface";
import { Button } from "@/components/ui/button";
import { LogOut, Plus, History, ChevronLeft, Trash2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useState, useEffect } from "react";
import { formatDistanceToNow } from 'date-fns';
import { cn } from "@/lib/utils";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface SessionInfo {
  id: string;
  timestamp: Date;
}

export default function HomeContent() {
  const { logout } = useAuth();
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(undefined);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  // Load sessions from localStorage on component mount
  useEffect(() => {
    const savedSessions = localStorage.getItem('carepay_sessions');
    if (savedSessions) {
      try {
        // Parse the sessions and convert timestamp strings back to Date objects
        const parsedSessions = JSON.parse(savedSessions).map((session: any) => ({
          ...session,
          timestamp: new Date(session.timestamp)
        }));
        setSessions(parsedSessions);
      } catch (error) {
        console.error('Error loading sessions from localStorage:', error);
      }
    }
  }, []);

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('carepay_sessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  const handleCreateNewSession = () => {
    // If there's a current session, save it to the sessions list
    if (currentSessionId) {
      setSessions(prev => {
        // Check if this session is already in the list
        if (!prev.some(s => s.id === currentSessionId)) {
          return [...prev, {
            id: currentSessionId,
            timestamp: new Date()
          }];
        }
        return prev;
      });
    }
    
    // Reset the current session ID to trigger creation of a new session
    setCurrentSessionId(undefined);
    // Close the history drawer if it's open
    setIsHistoryOpen(false);
  };
  
  const handleSessionSelect = (sessionId: string) => {
    // If the current session is not yet saved, save it first
    if (currentSessionId && !sessions.some(s => s.id === currentSessionId)) {
      setSessions(prev => [...prev, {
        id: currentSessionId,
        timestamp: new Date()
      }]);
    }
    
    // Set the selected session as current
    setCurrentSessionId(sessionId);
    // Close the history drawer
    setIsHistoryOpen(false);
  };

  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    // Prevent the click from triggering session selection
    e.stopPropagation();
    
    // Remove the session from the list
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    
    // If the deleted session is the current one, create a new session
    if (currentSessionId === sessionId) {
      setCurrentSessionId(undefined);
    }
  };
  
  const handleLogout = () => {
    setShowLogoutConfirm(true);
  };

  const handleConfirmLogout = async () => {
    try {
      await logout();
      // The logout function from useAuth will handle the redirect
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
      {/* Mobile phone frame */}
      <div className="relative w-full max-w-md h-[700px] bg-white rounded-[40px] shadow-xl overflow-hidden border-8 border-gray-800">
        {/* Phone notch */}
        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1/3 h-6 bg-gray-800 rounded-b-xl z-10"></div>
        
        {/* Phone header */}
        <header className="bg-white border-b border-border py-3 px-4 sticky top-0 z-10 shadow-sm mt-6">
          <div className="flex items-center justify-between">
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
            
            {/* Action buttons */}
            <div className="flex items-center gap-2">
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 px-2 text-xs"
                onClick={() => setIsHistoryOpen(prev => !prev)}
                title={isHistoryOpen ? "Close History" : "Session History"}
              >
                <History size={16} />
              </Button>
              
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 px-2 text-xs"
                onClick={handleCreateNewSession}
                title="New Chat"
              >
                <Plus size={16} className="mr-1" />
                New Chat
              </Button>
              
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 px-2 text-xs text-red-600 hover:text-red-700 hover:bg-red-50"
                onClick={handleLogout}
                title="Logout"
              >
                <LogOut size={16} />
              </Button>
            </div>
          </div>
        </header>
        
        {/* Session history drawer */}
        <div className={cn(
          "absolute inset-y-[72px] left-0 w-3/4 bg-white border-r border-border z-20 transition-transform duration-300 transform",
          isHistoryOpen ? "translate-x-0" : "-translate-x-full"
        )}>
          <div className="flex flex-col h-full">
            <div className="p-3 border-b border-border flex justify-between items-center">
              <h2 className="font-semibold">Session History</h2>
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 w-8 p-0"
                onClick={() => setIsHistoryOpen(false)}
              >
                <ChevronLeft size={16} />
              </Button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-2">
              {sessions.length > 0 ? (
                <div className="space-y-2">
                  {sessions.map((session) => (
                    <Button
                      key={session.id}
                      variant={currentSessionId === session.id ? "secondary" : "outline"}
                      className="w-full justify-start text-left h-auto py-2 pr-2"
                      onClick={() => handleSessionSelect(session.id)}
                    >
                      <div className="flex justify-between items-center w-full">
                        <div className="flex flex-col items-start">
                          <span className="text-xs font-medium truncate w-full">
                            Session #{session.id.substring(0, 8)}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {formatDistanceToNow(session.timestamp, { addSuffix: true })}
                          </span>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                          onClick={(e) => handleDeleteSession(session.id, e)}
                          title="Delete session"
                        >
                          <Trash2 size={14} />
                        </Button>
                      </div>
                    </Button>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4 text-center">
                  <History size={24} className="mb-2 opacity-50" />
                  <p className="text-sm">No previous sessions</p>
                  <p className="text-xs">Start a new chat to begin</p>
                </div>
              )}
            </div>
            
            <div className="p-3 border-t border-border">
              <Button
                variant="outline"
                className="w-full"
                onClick={handleCreateNewSession}
              >
                <Plus size={16} className="mr-1" />
                New Session
              </Button>
            </div>
          </div>
        </div>
        
        <div className="h-[calc(100%-106px)] overflow-hidden">
          <ChatInterface 
            key={currentSessionId || "new-session"} 
            onSessionCreated={setCurrentSessionId}
            sessionId={currentSessionId}
          />
        </div>
        
        {/* Phone home indicator */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 w-1/3 h-1 bg-gray-800 rounded-full"></div>
      </div>
      
      {/* Logout confirmation dialog */}
      <AlertDialog open={showLogoutConfirm} onOpenChange={setShowLogoutConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure you want to logout?</AlertDialogTitle>
            <AlertDialogDescription>
              You will need to login again to access the application.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmLogout} className="bg-red-600 hover:bg-red-700">
              Logout
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
} 