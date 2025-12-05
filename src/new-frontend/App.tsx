import React, { useState, useRef } from 'react';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import AnalysisResult from './components/AnalysisResult';
import { evaluateBite, askRag, extractAdvice } from './services/dataService';
import { AppView, BiteAnalysis, ChatMessage } from './types';

const App: React.FC = () => {
  const [view, setView] = useState<AppView>(AppView.HOME);
  const [analysis, setAnalysis] = useState<BiteAnalysis | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Backend state tracking
  const [hasPrediction, setHasPrediction] = useState(false);
  const [lastUserMessage, setLastUserMessage] = useState("");
  const evalResRef = useRef<any>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [remainingOptions, setRemainingOptions] = useState<string[]>([]);

  const allOptions = [
    "Relief & treatment",
    "Prevention tips",
    "About this insect",
  ];

  const addMessage = (sender: 'user' | 'ranger', text: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        sender,
        text,
        timestamp: new Date()
      }
    ]);
  };

  // INITIAL ANALYSIS
  const handleAnalyze = async (imageBase64: string, notes: string) => {
    setView(AppView.ANALYZING);
    setUploadedImage(imageBase64);
    setError(null);
    setLastUserMessage(notes);
    setHistory((prev) => [...prev, notes]);

    try {
      const evalResult = await evaluateBite({
        user_text: notes,
        image_gcs_uri: null,
        image_base64: imageBase64,
        first_call: true,
        history: history,
      });

      if (evalResult.needs_fix) {
        setError(evalResult.error || "For best results, please describe your symptoms and where you were when you got bitten");
        setView(AppView.HOME);
        return;
      }

      const passed = evalResult.eval;
      evalResRef.current = passed;

      const pred = passed?.prediction || "unknown";
      const conf = typeof passed?.confidence === "number" ? passed.confidence : 0;

      const scientificNameMap: Record<string, string> = {
        ants: 'Formicidae family',
        bed_bugs: 'Cimex lectularius family',
        chiggers: 'Trombiculidae family',
        fleas: 'Siphonaptera family',
        mosquitos: 'Culicidae family',
        spiders: 'Araneae family',
        ticks: 'Ixodida family',
      };

      const prettyName = pred
        .replaceAll('_', ' ')
        .replace(/s$/, '')
        .replace(/^./, str => str.toUpperCase());

      const baseDangerLevelMap: Record<string, 'Low' | 'Moderate' | 'High'> = {
        ants: 'Low',
        bed_bugs: 'Low',
        chiggers: 'Low',
        fleas: 'Low',
        mosquitos: 'Moderate',   // disease risk (e.g., West Nile, dengue) [web:27][web:35][web:37]
        spiders: 'Moderate',     // some medically important species [web:21][web:24][web:25]
        ticks: 'High',           // Lyme and other serious diseases [web:26][web:29][web:32][web:34][web:40]
      };

      const dangerLevel = baseDangerLevelMap[pred] ?? 'Low';

      const biteAnalysis: BiteAnalysis = {
        bugName: prettyName,
        scientificName: scientificNameMap[pred] ?? prettyName,
        description: `Based on the image and your description, this appears to be a ${prettyName.toLowerCase()} bite with ${Math.round(conf * 100)}% confidence. These are common outdoor pests that can cause irritation.`,
        dangerLevel,
      };

      setAnalysis(biteAnalysis);
      setView(AppView.RESULT);

      // Add initial greeting
      addMessage(
        'ranger',
        `Howdy! Looks like you might have had a run-in with a ${biteAnalysis.bugName}. I've got your analysis ready. Do you have any specific questions for me?`
      );

      setRemainingOptions(allOptions);
      setShowSuggestions(true);
      setHasPrediction(true);
    } catch (err) {
      console.error(err);
      setError("Oops, something went wrong while contacting the server.");
      setView(AppView.HOME);
    }
  };

  // HANDLE FOLLOW-UP MESSAGES
  const handleSendMessage = async (text: string) => {
    if (!hasPrediction || !evalResRef.current) return;

    addMessage('user', text);
    setIsChatLoading(true);
    setShowSuggestions(false);

    try {
      // Check if it's a courtesy message or irrelevant question
      const relevanceCheck = await evaluateBite({
        user_text: text,
        image_gcs_uri: null,
        image_base64: null,
        first_call: false,
        history: [],
      });

      if (relevanceCheck.eval?.courtesy) {
        addMessage('ranger', "You're welcome, partner! Let me know if you have other questions about the bite. Stay safe on the trail!");
        setIsChatLoading(false);
        return;
      }

      if (!relevanceCheck.eval?.question_relevant) {
        const errorMsg = relevanceCheck.eval?.improve_message ||
          "I can only answer questions about insect bites, symptoms, prevention, or treatment. Keep it trail-related!";
        addMessage('ranger', errorMsg);
        setIsChatLoading(false);
        return;
      }

      // Question is relevant → call RAG
      const ragRes = await askRag({
        question: text,
        symptoms: lastUserMessage,
        bug_class: evalResRef.current?.prediction || "unknown",
        conf: evalResRef.current?.confidence || 0.0,
      });

      const answer = extractAdvice(ragRes.llm);
      addMessage('ranger', answer);

      // Update remaining suggestions if user clicked one
      if (allOptions.includes(text)) {
        const nextOptions = remainingOptions.filter((o) => o !== text);
        setRemainingOptions(nextOptions);
        if (nextOptions.length > 0) {
          setTimeout(() => setShowSuggestions(true), 400);
        }
      }

    } catch (err) {
      console.error(err);
      addMessage('ranger', "My radio signal is breaking up. Can you try asking that again?");
    } finally {
      setIsChatLoading(false);
    }
  };

  const resetApp = () => {
    setAnalysis(null);
    setUploadedImage('');
    setMessages([]);
    setView(AppView.HOME);
    setError(null);
    setHasPrediction(false);
    setLastUserMessage('');
    evalResRef.current = null;
    setHistory([]);
    setShowSuggestions(false);
    setRemainingOptions([]);
  };

  return (
    <div className="min-h-screen bg-earth-50 flex flex-col font-sans">
      <Header />

      <main className="flex-grow p-4 md:p-8">
        {error && (
          <div className="max-w-2xl mx-auto mb-6 bg-red-100 border border-red-300 text-red-800 px-4 py-3 rounded relative" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {view === AppView.HOME && (
          <div className="max-w-4xl mx-auto text-center space-y-6 mt-8">
            <div className="space-y-3">
              <h2 className="text-4xl font-serif font-bold text-earth-900">What Bit You?</h2>
              <p className="text-lg text-earth-600">
                Upload a photo and describe the scene. We'll play detective and help you heal up!
              </p>
            </div>
            <UploadSection onAnalyze={handleAnalyze} isAnalyzing={false} />
          </div>
        )}

        {view === AppView.ANALYZING && (
          <div className="max-w-2xl mx-auto mt-20 text-center">
            <div className="relative w-32 h-32 mx-auto mb-8">
               <div className="absolute inset-0 border-4 border-earth-200 rounded-full"></div>
               <div className="absolute inset-0 border-4 border-forest-500 rounded-full border-t-transparent animate-spin"></div>
               <div className="absolute inset-0 flex items-center justify-center text-4xl">🔍</div>
            </div>
            <h2 className="text-2xl font-serif font-bold text-earth-900 mb-2">Analyzing your bite...</h2>
            <p className="text-earth-600">Ranger Rick is flipping through his field guide.</p>
          </div>
        )}

        {view === AppView.RESULT && analysis && (
          <div className="max-w-6xl mx-auto mt-8">
            {/* Top-left Start Over button */}
            <button
              onClick={resetApp}
              className="mb-4 flex items-center text-earth-600 hover:text-forest-700 transition-colors font-medium"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={2}
                stroke="currentColor"
                className="w-4 h-4 mr-1"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18"
                />
              </svg>
              Start Over
            </button>

            <AnalysisResult
              analysis={analysis}
              uploadedImage={uploadedImage}
              messages={messages}
              onSendMessage={handleSendMessage}
              isChatLoading={isChatLoading}
              showSuggestions={showSuggestions}
              suggestions={remainingOptions}
              onSuggestionClick={handleSendMessage}
            />
          </div>
        )}

      </main>

      <footer className="bg-earth-200 text-earth-800 py-6 mt-auto">
        <div className="max-w-5xl mx-auto px-4 text-center">
          <p className="text-sm font-medium">© {new Date().getFullYear()} BiteFinder. Built for the wild.</p>
          <p className="text-xs text-earth-600 mt-2">Always seek professional medical attention for serious reactions.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
