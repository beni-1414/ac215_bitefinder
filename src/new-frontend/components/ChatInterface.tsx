import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../types';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  showSuggestions?: boolean;
  suggestions?: string[];
  onSuggestionClick?: (suggestion: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  showSuggestions = false,
  suggestions = [],
  onSuggestionClick
}) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, showSuggestions]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-xl shadow-xl overflow-hidden border border-earth-200">
      {/* Chat Header */}
      <div className="bg-forest-800 p-5 flex items-center gap-3 border-b border-forest-900">
        <div className="relative">
          <div className="w-10 h-10 rounded-full bg-earth-200 flex items-center justify-center border-2 border-forest-400 overflow-hidden">
            <span className="text-xl">🤠</span>
          </div>
          <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-400 border-2 border-forest-800 rounded-full"></div>
        </div>
        <div>
          <h3 className="text-white font-serif font-bold text-lg">Ranger Rick</h3>
          <p className="text-forest-200 text-sm">Wilderness Expert • Online</p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-earth-50">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`
                max-w-[80%] rounded-2xl p-4 text-base leading-relaxed shadow-sm
                ${msg.sender === 'user'
                  ? 'bg-forest-600 text-white rounded-tr-none'
                  : 'bg-white text-earth-900 border border-earth-200 rounded-tl-none'
                }
              `}
            >
              {msg.text}
              <div className={`text-xs mt-1 ${msg.sender === 'user' ? 'text-forest-200' : 'text-earth-400'}`}>
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-earth-200 rounded-2xl rounded-tl-none p-3 shadow-sm flex gap-1 items-center">
              <div className="w-2 h-2 bg-earth-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-earth-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-earth-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
            </div>
          </div>
        )}

        {/* Question Suggestions */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="flex flex-col gap-2 pt-2">
            <p className="text-sm text-earth-600 font-medium">What would you like to know?</p>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((option, idx) => (
                <button
                  key={idx}
                  onClick={() => onSuggestionClick?.(option)}
                  disabled={isLoading}
                  className="bg-forest-100 hover:bg-forest-200 text-forest-800 border border-forest-300 rounded-lg px-5 py-3 text-base font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="p-4 bg-white border-t border-earth-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask Ranger Rick a question..."
            className="flex-1 rounded-full border-earth-300 bg-earth-50 px-5 py-3 focus:ring-2 focus:ring-forest-500 focus:border-forest-500 outline-none transition-all text-base text-earth-900"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className={`
              p-3 rounded-full text-white transition-all
              ${!inputText.trim() || isLoading
                ? 'bg-earth-300'
                : 'bg-forest-600 hover:bg-forest-700'
              }
            `}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 transform rotate-90">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
