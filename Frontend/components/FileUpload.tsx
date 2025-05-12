"use client";

import { useState, useRef } from "react";
import { UploadCloud, X, FileImage, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface FileUploadProps {
  onUpload: (file: File, documentType: 'aadhaar' | 'pan') => void;
  onCancel: () => void;
}

export default function FileUpload({ onUpload, onCancel }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<'aadhaar' | 'pan' | ''>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };
  
  const handleUpload = () => {
    if (selectedFile && documentType) {
      onUpload(selectedFile, documentType);
      setSelectedFile(null);
      setDocumentType('');
    }
  };
  
  const handleCancel = () => {
    setSelectedFile(null);
    setDocumentType('');
    onCancel();
  };
  
  return (
    <div className="p-3">
      <div 
        className={cn(
          "border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center transition-all",
          dragActive ? "border-blue-500 bg-blue-50" : "border-border",
          selectedFile ? "bg-slate-50" : "bg-transparent"
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {selectedFile ? (
          <div className="flex flex-col items-center space-y-3 w-full">
            <div className="flex items-center gap-2 p-2 bg-white rounded-lg border border-border w-full">
              {selectedFile.type.includes("image") ? (
                <FileImage size={18} className="text-blue-500" />
              ) : (
                <FileText size={18} className="text-blue-500" />
              )}
              
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{selectedFile.name}</p>
                <p className="text-xs text-slate-500">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
              
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSelectedFile(null)}
                className="h-6 w-6"
              >
                <X size={14} />
              </Button>
            </div>
            
            <div className="w-full">
              <Select 
                value={documentType} 
                onValueChange={(value: 'aadhaar' | 'pan' | '') => setDocumentType(value)}
              >
                <SelectTrigger className="w-full text-sm">
                  <SelectValue placeholder="Select document type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="aadhaar">Aadhaar Card</SelectItem>
                  <SelectItem value="pan">PAN Card</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => setSelectedFile(null)}>
                Change
              </Button>
              <Button size="sm" onClick={handleUpload} disabled={!documentType}>
                Upload
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-3">
              <UploadCloud size={20} className="text-blue-500" />
            </div>
            <h3 className="text-base font-semibold mb-1">Upload Document</h3>
            <p className="text-xs text-slate-500 text-center mb-3">
              Drag and drop your document here or click to browse
            </p>
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              size="sm"
              className="mb-1"
            >
              Browse Files
            </Button>
            <p className="text-xs text-slate-500">
              Supports: Aadhaar Card, PAN Card
            </p>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleChange}
              accept="image/*,.pdf"
              className="hidden"
            />
          </>
        )}
      </div>
      
      {!selectedFile && (
        <div className="flex justify-end mt-2">
          <Button variant="ghost" size="sm" onClick={handleCancel}>Cancel</Button>
        </div>
      )}
    </div>
  );
}