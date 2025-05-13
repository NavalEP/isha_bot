import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, X } from "lucide-react";

interface ChatSession {
  id: string;
  timestamp: Date;
  preview: string;
}

interface ChatHistoryProps {
  sessions: ChatSession[];
  onSelectSession: (sessionId: string) => void;
  onClose: () => void;
}

export default function ChatHistory({ sessions, onSelectSession, onClose }: ChatHistoryProps) {
  return (
    <div className="absolute inset-0 bg-white z-20 flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="text-lg font-medium">Chat History</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X size={18} />
        </Button>
      </div>
      
      <ScrollArea className="flex-1">
        {sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-4">
            <MessageSquare className="h-12 w-12 text-slate-300 mb-2" />
            <p className="text-slate-500">No previous chats found</p>
          </div>
        ) : (
          <div className="p-4 space-y-2">
            {sessions.map((session) => (
              <Button
                key={session.id}
                variant="outline"
                className="w-full justify-start h-auto py-3 px-4"
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex flex-col items-start text-left">
                  <span className="text-xs text-slate-500">
                    {session.timestamp.toLocaleString()}
                  </span>
                  <span className="line-clamp-1 text-sm mt-1">{session.preview}</span>
                </div>
              </Button>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
} 