import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageCircle, X, Send, Globe, AlertCircle } from "lucide-react";
import { Card } from "@/components/ui/card";

interface Message {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: Date;
  language?: string;
  confidence?: number;
  sources?: string[];
}

interface ChatResponse {
  response: string;
  confidence: number;
  response_time: number;
  cached: boolean;
  detected_language: string;
  detected_intent: string;
  sources?: string[];
}

const languages = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'hi', name: 'हिंदी', flag: '🇮🇳' },
  { code: 'ur', name: 'اردو', flag: '🇵🇰' },
];

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AKGEC assistant. How can I help you today?',
      isBot: true,
      timestamp: new Date(),
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState(languages[0]);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Backend API configuration - Using import.meta.env for Vite
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8081';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Close chatbot when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (chatContainerRef.current && !chatContainerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // API call to Python backend
  const callBackendAPI = async (message: string, language: string): Promise<ChatResponse> => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        user_id: 'frontend_user',
        language: language === 'auto' ? null : language
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isBot: false,
      timestamp: new Date(),
      language: selectedLanguage.code
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);
    setError(null);

    try {
      // Call actual Python backend API
      const apiResponse = await callBackendAPI(currentInput, selectedLanguage.code);
      
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: apiResponse.response,
        isBot: true,
        timestamp: new Date(),
        language: apiResponse.detected_language,
        confidence: apiResponse.confidence,
        sources: apiResponse.sources
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('API call failed:', error);
      
      // Fallback response when API fails
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: selectedLanguage.code === 'hi' 
          ? "माफ करें, अभी सर्वर से कनेक्ट नहीं हो पा रहा। कृपया बाद में कोशिश करें।"
          : "Sorry, I'm having trouble connecting to the server. Please try again later.",
        isBot: true,
        timestamp: new Date(),
        language: selectedLanguage.code
      };

      setMessages(prev => [...prev, errorMessage]);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setIsTyping(false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  // Test backend connection
  const testConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setError(null);
        console.log('Backend connection successful');
      } else {
        setError('Backend connection failed');
      }
    } catch (error) {
      setError('Cannot connect to backend server');
      console.error('Connection test failed:', error);
    }
  };

  // Test connection on mount
  useEffect(() => {
    if (isOpen) {
      testConnection();
    }
  }, [isOpen]);

  return (
    <div className="fixed bottom-6 right-6 z-50" ref={chatContainerRef}>
      {/* Chat Window */}
      {isOpen && (
        <Card className={`mb-4 w-80 h-96 glass-morphic border border-white/8 shadow-2xl overflow-hidden ${
          isOpen ? 'chatbot-enter' : 'chatbot-exit'
        }`}>
          {/* Header */}
          <div className="bg-gradient-to-r from-primary/90 to-primary-hover/90 backdrop-blur-sm text-primary-foreground p-4 flex items-center justify-between border-b border-white/10">
            <div className="flex items-center gap-2">
              <MessageCircle className="w-5 h-5" />
              <span className="font-semibold">AKGEC Assistant</span>
              {error && (
                <AlertCircle className="w-4 h-4 text-red-300" title={`Error: ${error}`} />
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Language Selector */}
              <select
                value={selectedLanguage.code}
                onChange={(e) => setSelectedLanguage(languages.find(l => l.code === e.target.value) || languages[0])}
                className="bg-white/10 backdrop-blur-sm border border-white/20 rounded px-2 py-1 text-xs transition-all duration-300 hover:bg-white/20"
              >
                {languages.map(lang => (
                  <option key={lang.code} value={lang.code} className="text-primary bg-white">
                    {lang.flag} {lang.name}
                  </option>
                ))}
              </select>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="h-6 w-6 text-primary-foreground hover:bg-white/20 transition-all duration-300 button-hover"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 p-4 overflow-y-auto bg-white/3 backdrop-blur-md h-64 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-3 flex ${message.isBot ? 'justify-start' : 'justify-end'}`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg transition-all duration-300 ${
                    message.isBot
                      ? 'bg-white/15 text-foreground backdrop-blur-md border border-white/10 shadow-lg'
                      : 'bg-primary/80 text-primary-foreground backdrop-blur-sm shadow-lg'
                  }`}
                >
                  <p className="text-sm">{message.text}</p>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-xs opacity-70">
                      {formatTime(message.timestamp)}
                    </span>
                    {message.confidence && (
                      <span className="text-xs opacity-70">
                        {Math.round(message.confidence * 100)}%
                      </span>
                    )}
                  </div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="text-xs opacity-60 mt-1">
                      Sources: {message.sources.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="mb-3 flex justify-start">
                <div className="bg-white/15 backdrop-blur-md border border-white/10 p-3 rounded-lg shadow-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 bg-white/5 backdrop-blur-md border-t border-white/10">
            <div className="flex gap-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={`Ask about AKGEC... (${selectedLanguage.name})`}
                className="flex-1 bg-white/10 backdrop-blur-sm border-white/20 text-foreground placeholder:text-foreground/50 transition-all duration-300 focus:bg-white/15 focus:border-white/30"
                onKeyPress={(e) => e.key === 'Enter' && !isTyping && handleSendMessage()}
                disabled={isTyping}
              />
              <Button
                onClick={handleSendMessage}
                size="icon"
                className="bg-primary/80 hover:bg-primary-hover/80 backdrop-blur-sm transition-all duration-300 button-hover shadow-lg"
                disabled={isTyping || !inputText.trim()}
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
            {error && (
              <div className="text-xs text-red-300 mt-2 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                Connection issue: Using offline mode
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Toggle Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-14 h-14 rounded-full bg-gradient-to-r from-primary/90 to-accent/90 hover:from-primary hover:to-accent shadow-2xl transition-all duration-500 backdrop-blur-sm border border-white/10 ${
          isOpen ? 'rotate-45 scale-110' : 'bounce-in hover:scale-110'
        } button-hover`}
        size="icon"
      >
        {isOpen ? (
          <X className="w-6 h-6 text-white transition-transform duration-300" />
        ) : (
          <MessageCircle className="w-6 h-6 text-white transition-transform duration-300" />
        )}
      </Button>
    </div>
  );
};

export default ChatBot;