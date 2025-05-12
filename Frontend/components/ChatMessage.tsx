import { Check, FileImage, FileText, User, Bot, CreditCard, ArrowRight } from "lucide-react";
import { Message, MessageType, MessageAttachment, BureauDecision, EMIPlan } from "@/types/chat";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isBot = message.type === 'bot';
  const formattedTime = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  }).format(message.timestamp);
  
  return (
    <div className={cn(
      "flex gap-2 max-w-[85%]",
      isBot ? "self-start" : "self-end ml-auto"
    )}>
      {isBot && (
        <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center shrink-0 mt-1">
          <Bot size={14} className="text-green-600" />
        </div>
      )}
      
      <div className={cn(
        "flex flex-col gap-1",
        !isBot && "items-end"
      )}>
        <div className="flex items-center gap-1">
          <span className="text-xs font-medium text-slate-700">
            {isBot ? "Isha" : "You"}
          </span>
          <span className="text-xs text-slate-500">{formattedTime}</span>
        </div>
        
        <div className={cn(
          "px-3 py-2 rounded-2xl text-sm",
          isBot ? "bg-slate-100 text-slate-900 rounded-tl-none" : "bg-blue-600 text-white rounded-tr-none"
        )}>
          <p className="whitespace-pre-wrap">{message.text}</p>
        </div>
        
        {/* Bureau Decision Display */}
        {isBot && message.bureauDecision && (
          <div className="mt-2 border border-slate-200 rounded-lg overflow-hidden bg-white">
            <div className="bg-slate-50 border-b border-slate-200 p-3">
              <h3 className="font-medium text-sm">Loan Decision</h3>
              <div className="mt-1 space-y-1">
                {message.bureauDecision.status && (
                  <div className="flex items-start text-xs">
                    <span className="font-medium mr-2 text-slate-600">Status:</span>
                    <span className={cn(
                      "font-medium",
                      message.bureauDecision.status === "APPROVED" ? "text-green-600" : 
                      message.bureauDecision.status === "REJECTED" ? "text-red-600" : "text-amber-600"
                    )}>
                      {message.bureauDecision.status.replace(/_/g, ' ')}
                    </span>
                  </div>
                )}
                {message.bureauDecision.reason && (
                  <div className="flex items-start text-xs">
                    <span className="font-medium mr-2 text-slate-600">Reason:</span>
                    <span>{message.bureauDecision.reason}</span>
                  </div>
                )}
                {message.bureauDecision.maxEligibleEMI && (
                  <div className="flex items-start text-xs">
                    <span className="font-medium mr-2 text-slate-600">Max Eligible EMI:</span>
                    <span>₹{Number(message.bureauDecision.maxEligibleEMI).toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>
            
            {message.bureauDecision.emiPlans && message.bureauDecision.emiPlans.length > 0 && (
              <div className="p-3">
                <h4 className="text-xs font-medium mb-2">Available Loan Plans</h4>
                <div className="space-y-2">
                  {message.bureauDecision.emiPlans.map((plan, index) => (
                    <EMIPlanCard key={index} plan={plan} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-col gap-1 mt-1 w-full">
            {message.attachments.map((attachment, index) => (
              <AttachmentPreview key={index} attachment={attachment} />
            ))}
          </div>
        )}
      </div>
      
      {!isBot && (
        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center shrink-0 mt-1">
          <User size={14} className="text-blue-600" />
        </div>
      )}
    </div>
  );
}

// EMI Plan Card Component
function EMIPlanCard({ plan }: { plan: EMIPlan }) {
  return (
    <div className="border border-slate-200 rounded-md p-2 bg-slate-50 hover:bg-slate-100 transition-colors">
      <div className="flex justify-between items-center">
        <div className="flex items-center">
          <CreditCard className="h-4 w-4 mr-2 text-blue-600" />
          <span className="font-medium text-xs">{plan.planName} Plan</span>
        </div>
        {plan.creditLimit && (
          <span className="text-xs font-medium text-green-600">
            ₹{Number(plan.creditLimit).toLocaleString()}
          </span>
        )}
      </div>
      
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        {plan.emi && (
          <div>
            <span className="text-slate-500">EMI:</span>
            <span className="font-medium ml-1">₹{Number(plan.emi).toLocaleString()}</span>
          </div>
        )}
        {plan.downPayment && plan.downPayment !== "0" && (
          <div>
            <span className="text-slate-500">Down:</span>
            <span className="font-medium ml-1">₹{Number(plan.downPayment).toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function AttachmentPreview({ attachment }: { attachment: MessageAttachment }) {
  const isPdf = attachment.type === "document" && attachment.name.endsWith(".pdf");
  const isImage = attachment.type === "image";
  
  return (
    <div className="flex items-center gap-2 p-2 rounded-md bg-white border border-border max-w-full">
      {isImage ? (
        <div className="relative overflow-hidden rounded-md h-16 w-16 shrink-0">
          <Image
            src={attachment.url}
            alt={attachment.name}
            fill
            className="object-cover"
          />
        </div>
      ) : (
        <div className="w-8 h-8 rounded-md bg-slate-100 flex items-center justify-center shrink-0">
          {isPdf ? (
            <FileText size={16} className="text-red-500" />
          ) : (
            <FileImage size={16} className="text-blue-500" />
          )}
        </div>
      )}
      
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{attachment.name}</p>
        <p className="text-xs text-slate-500">
          {isImage ? "Image" : "Document"}
        </p>
      </div>
      
      <div className="w-5 h-5 rounded-full bg-green-100 flex items-center justify-center shrink-0">
        <Check size={12} className="text-green-600" />
      </div>
    </div>
  );
}