import { LogOut, MessageSquarePlus, History } from "lucide-react";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import { useAuth } from "@/hooks/useAuth";
import { useRouter } from "next/navigation";

interface AppHeaderProps {
  onNewChat: () => void;
  onViewHistory: () => void;
}

export default function AppHeader({ onNewChat, onViewHistory }: AppHeaderProps) {
  const { logout } = useAuth();
  const router = useRouter();
  
  const handleLogout = () => {
    logout();
  };

  return (
    <header className="bg-white border-b border-border py-3 px-4 sticky top-0 z-10 shadow-sm">
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
        
        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={onNewChat}
            title="New Chat"
          >
            <MessageSquarePlus size={18} />
          </Button>
          
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={onViewHistory}
            title="Chat History"
          >
            <History size={18} />
          </Button>
          
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={handleLogout}
            title="Logout"
            className="text-red-500"
          >
            <LogOut size={18} />
          </Button>
        </div>
      </div>
    </header>
  );
} 