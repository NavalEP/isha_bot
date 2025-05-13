export type MessageType = 'user' | 'bot';

export interface Message {
  id: string;
  type: MessageType;
  text: string;
  timestamp: Date;
  attachments?: MessageAttachment[];
  bureauDecision?: BureauDecision;
  sessionExpired?: boolean;
}

export interface MessageAttachment {
  type: 'image' | 'document';
  name: string;
  url: string;
}

// EMI Plan interface
export interface EMIPlan {
  planName: string;
  creditLimit: string | null;
  emi: string | null;
  downPayment: string | null;
}

// Bureau Decision interface
export interface BureauDecision {
  status: string | null;
  reason: string | null;
  maxEligibleEMI: string | null;
  emiPlans: EMIPlan[];
}

// Chat Session interface for history
export interface ChatSession {
  id: string;
  timestamp: Date;
  preview: string;
}

// Response types for API
export interface ChatResponse {
  status: string;
  session_id: string;
  response: string;
  progress?: number;
  message?: string;
  bureauDecision?: BureauDecision;
}