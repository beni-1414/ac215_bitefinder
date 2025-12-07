import React from 'react';
import { BiteAnalysis, ChatMessage } from '../types';
import ChatInterface from './ChatInterface';

interface AnalysisResultProps {
  analysis: BiteAnalysis;
  uploadedImage: string;
  onReset: () => void;
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isChatLoading: boolean;
  showSuggestions?: boolean;
  suggestions?: string[];
  onSuggestionClick?: (suggestion: string) => void;
  onNavigateToGuide?: () => void;
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({
  analysis,
  uploadedImage,
  onReset,
  messages,
  onSendMessage,
  isChatLoading,
  showSuggestions = false,
  suggestions = [],
  onSuggestionClick,
  onNavigateToGuide
}) => {

  const getDangerColor = (level: string) => {
    switch (level) {
      case 'Emergency':
      case 'High': return 'bg-red-100 text-red-800 border-red-300';
      case 'Moderate': return 'bg-amber-100 text-amber-800 border-amber-300';
      default: return 'bg-green-100 text-green-800 border-green-300';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">

      {/* Left Column: Chat Interface - MAIN FOCUS */}
      <div className="lg:col-span-2">
        <button
          onClick={onReset}
          className="flex items-center text-earth-600 hover:text-forest-700 transition-colors font-medium mb-4"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4 mr-1">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
          Start Over
        </button>

        <div className="mb-4">
          <h2 className="text-2xl font-serif font-bold text-earth-900">Ask Ranger Rick</h2>
          <p className="text-sm text-earth-600">Get personalized advice about your {analysis.bugName} bite</p>
        </div>

        <ChatInterface
          messages={messages}
          onSendMessage={onSendMessage}
          isLoading={isChatLoading}
          showSuggestions={showSuggestions}
          suggestions={suggestions}
          onSuggestionClick={onSuggestionClick}
        />
      </div>

      {/* Right Column: Simplified Result Card - SIDEBAR */}
      <div className="lg:col-span-1">
        <div className="lg:sticky lg:top-24">
          <div className="mb-3">
            <h3 className="text-lg font-serif font-bold text-earth-900">Your Bite Analysis</h3>
          </div>

          <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-earth-200">
            <div className="relative h-48 bg-earth-200">
              <img
                src={`data:image/jpeg;base64,${uploadedImage}`}
                alt="Uploaded bite"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end">
                <div className="p-4 text-white">
                  <p className="text-xs font-medium opacity-90 uppercase tracking-wider">Identified as</p>
                  <h1 className="text-2xl font-serif font-bold">{analysis.bugName}</h1>
                  <p className="italic text-earth-200 text-xs">{analysis.scientificName}</p>
                </div>
              </div>
            </div>

            <div className="p-4">
              {/* Danger Level Badge */}
              <div className="mb-3">
                <span className={`px-3 py-1.5 rounded-full text-xs font-bold border ${getDangerColor(analysis.dangerLevel)}`}>
                  {analysis.dangerLevel} Risk
                </span>
              </div>

              <p className="text-base text-earth-800 leading-relaxed mb-4">
                {analysis.description}
              </p>

              {/* Ranger Rick's Rules */}
              <div className="bg-forest-50 rounded-lg p-4 border border-forest-100 mb-3">
                <h3 className="font-serif font-bold text-forest-800 mb-3 text-base flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                  Ranger Rick's Rules
                </h3>
                <ul className="space-y-2">
                  <li className="text-base text-forest-900 flex items-start">
                    <span className="mr-2 text-forest-500 mt-1">🌲</span>
                    <span>Use insect repellent when venturing outdoors</span>
                  </li>
                  <li className="text-base text-forest-900 flex items-start">
                    <span className="mr-2 text-forest-500 mt-1">👕</span>
                    <span>Wear long sleeves and pants in wooded or grassy areas</span>
                  </li>
                  <li className="text-base text-forest-900 flex items-start">
                    <span className="mr-2 text-forest-500 mt-1">🔍</span>
                    <span>Check yourself for bites after outdoor activities</span>
                  </li>
                </ul>
              </div>

              {/* Prevention Guide Link */}
              {onNavigateToGuide && (
                <button
                  onClick={onNavigateToGuide}
                  className="w-full mb-3 bg-forest-600 hover:bg-forest-700 text-white font-bold py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <span className="text-xl">🛡️</span>
                  <span>Full Prevention Guide</span>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                  </svg>
                </button>
              )}

              <div className="mt-4 p-3 bg-orange-50 border border-orange-100 rounded-lg">
                <h4 className="font-bold text-orange-800 mb-1 text-xs uppercase">Medical Disclaimer</h4>
                <p className="text-xs text-orange-700">
                  BiteFinder is an AI tool and not a substitute for professional medical advice. If you experience severe symptoms, call emergency services immediately.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResult;
