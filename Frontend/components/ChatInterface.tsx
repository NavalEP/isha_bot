"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, RefreshCcw } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import ChatMessage from "@/components/ChatMessage";
import { Message, BureauDecision } from "@/types/chat";
import useApi from "@/hooks/useApi";

interface ChatInterfaceProps {
  initialMessage?: string;
  sessionId?: string | null;
  onNewChat?: () => void;
}

export default function ChatInterface({ 
  initialMessage = "Hello! I'm here to assist you with your healthcare loan application. Let's get started.\n\nFirst, could you please provide me with the following details?\n1. Your full name\n2. Your phone number\n3. The cost of the treatment you are seeking\n4. Your monthly income",
  sessionId = null,
  onNewChat
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "bot",
      text: initialMessage,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [applicationProgress, setApplicationProgress] = useState(10);
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(sessionId || undefined);
  const [connectionError, setConnectionError] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sessionInitializedRef = useRef<boolean>(false);
  const api = useApi();

  // Initialize chat session
  useEffect(() => {
    // Only initialize session if it hasn't been initialized yet
    if (!sessionInitializedRef.current && !currentSessionId) {
      sessionInitializedRef.current = true;
      initializeSession();
    }
  }, [currentSessionId]);

  // Update currentSessionId when sessionId prop changes
  useEffect(() => {
    if (sessionId) {
      setCurrentSessionId(sessionId);
    }
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const initializeSession = async () => {
    if (isProcessing || currentSessionId) return;
    
    setIsProcessing(true);
    setConnectionError(false);
    try {
      // Create a new session
      const sessionResponse = await api.createSession();
      
      if (sessionResponse.error) {
        // Check for connection issues
        if (sessionResponse.error.includes('fetch failed') || 
            sessionResponse.error.includes('NetworkError') ||
            sessionResponse.error.includes('ECONNREFUSED')) {
          setConnectionError(true);
          throw new Error('Could not connect to the chat server. Please check your connection and try again.');
        }
        throw new Error(sessionResponse.error);
      }
      
      if (sessionResponse.data?.session_id) {
        setCurrentSessionId(sessionResponse.data.session_id);
        console.log("Session initialized:", sessionResponse.data.session_id);
        // No need to send an initial message, the welcome message is already set
      }
    } catch (error) {
      console.error("Error initializing session:", error);
      sessionInitializedRef.current = false; // Reset flag to allow retrying
      
      const errorMessage: Message = {
        id: `error_init`,
        type: "bot",
        text: error instanceof Error 
          ? error.message 
          : "I'm sorry, there was an error starting the conversation. Please refresh and try again.",
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async () => {
    if (input.trim() === "") return;
    
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: "user",
      text: input,
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsProcessing(true);
    setConnectionError(false);
    
    try {
      const userInput = input;
      const response = await api.sendChatMessage(userInput, currentSessionId);
      
      if (response.error) {
        // Check for connection issues
        if (response.error.includes('fetch failed') || 
            response.error.includes('NetworkError') ||
            response.error.includes('ECONNREFUSED')) {
          setConnectionError(true);
          throw new Error('Could not connect to the chat server. Please check your connection and try again.');
        }
        throw new Error(response.error);
      }
      
      // Update session ID if it's returned
      if (response.data?.session_id) {
        setCurrentSessionId(response.data.session_id);
      }
      
      // Check if response contains JSON (for bureau decision) 
      let responseText = response.data?.response || "I received your message.";
      let bureauDecision: BureauDecision | undefined = undefined;
      let sessionExpired = false;
      
      // Check if session has expired
      if (responseText.includes("Session has expired after loan decision")) {
        sessionExpired = true;
        console.log("Session expired after loan decision");
      }
      
      try {
        // Try to parse the response as JSON
        const parsedResponse = JSON.parse(responseText);
        
        // Check if it has the expected structure with bureauDecision
        if (parsedResponse.message && parsedResponse.bureauDecision) {
          responseText = parsedResponse.message;
          bureauDecision = parsedResponse.bureauDecision;
          console.log("Detected bureau decision in response:", bureauDecision);
          
          // If we have a bureau decision, set progress to 100%
          setApplicationProgress(100);
        }
      } catch (parseError) {
        // Not JSON, use the string as is
        console.log("Response is not JSON, using as plain text");
      }
      
      const botMessage: Message = {
        id: `bot_${Date.now()}`,
        type: "bot",
        text: responseText,
        timestamp: new Date(),
        bureauDecision: bureauDecision,
        sessionExpired: sessionExpired
      };
      
      setMessages((prev) => [...prev, botMessage]);
      
      // Update progress if provided
      if (response.data?.progress) {
        setApplicationProgress(response.data.progress);
      } else if (!bureauDecision) {
        // Small progress increment on each message if not the final decision
        setApplicationProgress(prev => Math.min(prev + 5, 95));
      }
    } catch (error) {
      console.error("Error processing message:", error);
      
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        type: "bot",
        text: error instanceof Error 
          ? error.message 
          : "I'm sorry, there was an error processing your request. Please try again.",
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-50">
      {/* Progress bar */}
      <div className="w-full bg-slate-200 h-1">
        <div 
          className="bg-blue-600 h-1 transition-all duration-500 ease-in-out"
          style={{ width: `${applicationProgress}%` }}
        />
      </div>
      
      {/* Connection error message with retry button */}
      {connectionError && (
        <div className="bg-red-50 text-red-800 p-3 flex flex-col items-center justify-center space-y-2">
          <p className="text-sm">Unable to connect to the chat server.</p>
          <Button 
            onClick={initializeSession} 
            variant="outline" 
            className="text-xs" 
            size="sm"
            disabled={isProcessing}
          >
            <RefreshCcw size={14} className="mr-1" />
            Retry Connection
          </Button>
        </div>
      )}
      
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChatMessage message={message} onNewChat={onNewChat} />
            </motion.div>
          ))}
        </AnimatePresence>
        
        {/* Typing indicator */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center space-x-2 p-2 rounded-lg w-fit bg-slate-100"
          >
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div className="p-3 border-t border-border bg-white">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              className="pr-10 text-sm h-9 rounded-full"
              ref={inputRef}
              disabled={isProcessing}
            />
          </div>
          
          <Button
            onClick={handleSendMessage}
            disabled={!input.trim() || isProcessing}
            className="shrink-0 rounded-full w-9 h-9 p-0"
            aria-label="Send message"
          >
            <Send size={16} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1 text-center hidden sm:block">
          Press Enter to send, Shift + Enter for new line
        </p>
      </div>
    </div>
  );
}