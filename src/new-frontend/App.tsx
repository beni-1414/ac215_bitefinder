import React, { useState, useRef } from 'react';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import AnalysisResult from './components/AnalysisResult';
import PreventionGuide from './pages/PreventionGuide';
import AboutPage from './pages/AboutPage';
import SeasonalBugCalendar from './pages/SeasonalBugCalendar';
import BugEducation from './pages/BugEducation';
import { evaluateBite, askRag, extractAdvice, clearRagSession } from './services/dataService';
import { AppView, BiteAnalysis, ChatMessage } from './types';

const CHAT_STORAGE_KEY = 'bitefinder_chat_state';

const App: React.FC = () => {
  const [view, setView] = useState<AppView>(AppView.HOME);
  const [previousView, setPreviousView] = useState<AppView>(AppView.HOME);
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
  const [hasHydrated, setHasHydrated] = useState(false);

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

  // Rehydrate chat and analysis state on load
  React.useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = window.localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      setView(saved.view ?? AppView.HOME);
      setAnalysis(saved.analysis ?? null);
      setUploadedImage(saved.uploadedImage ?? '');
      setMessages(
        (saved.messages ?? []).map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp),
        }))
      );
      setHasPrediction(!!saved.hasPrediction);
      setLastUserMessage(saved.lastUserMessage ?? '');
      setHistory(saved.history ?? []);
      setRemainingOptions(saved.remainingOptions ?? []);
      setShowSuggestions(!!saved.showSuggestions);
      evalResRef.current = saved.evalResult ?? null;
    } catch (err) {
      console.warn('Unable to rehydrate chat state', err);
    } finally {
      setHasHydrated(true);
    }
  }, []);

  // Persist chat state after hydration
  React.useEffect(() => {
    if (!hasHydrated || typeof window === 'undefined') return;
    try {
      const payload = {
        view,
        analysis,
        uploadedImage,
        messages,
        hasPrediction,
        lastUserMessage,
        history,
        remainingOptions,
        showSuggestions,
        evalResult: evalResRef.current,
      };
      window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
      console.warn('Unable to persist chat state', err);
    }
  }, [hasHydrated, view, analysis, uploadedImage, messages, hasPrediction, lastUserMessage, history, remainingOptions, showSuggestions]);

  // INITIAL ANALYSIS
  const handleAnalyze = async (imageBase64: string, notes: string) => {
    clearRagSession();
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.removeItem(CHAT_STORAGE_KEY);
      } catch (err) {
        console.warn('Unable to clear chat state before new analysis', err);
      }
    }
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



      const biteAnalysis: BiteAnalysis = {
        bugName: prettyName,
        scientificName: scientificNameMap[pred] ?? prettyName,
        description: `Based on the image and your description, this appears to be a ${prettyName.toLowerCase()} bite with ${Math.round(conf * 100)}% confidence. These are common outdoor pests that can cause irritation.`,
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
      // // Check if it's a courtesy message or irrelevant question
      // const relevanceCheck = await evaluateBite({
      //   user_text: text,
      //   image_gcs_uri: null,
      //   image_base64: null,
      //   first_call: false,
      //   history: [],
      // });

      // if (relevanceCheck.eval?.courtesy) {
      //   addMessage('ranger', "You're welcome, partner! Let me know if you have other questions about the bite. Stay safe on the trail!");
      //   setIsChatLoading(false);
      //   return;
      // }

      // if (!relevanceCheck.eval?.question_relevant) {
      //   const errorMsg = relevanceCheck.eval?.improve_message ||
      //     "I can only answer questions about insect bites, symptoms, prevention, or treatment. Keep it trail-related!";
      //   addMessage('ranger', errorMsg);
      //   setIsChatLoading(false);
      //   return;
      // }

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
    clearRagSession();
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.removeItem(CHAT_STORAGE_KEY);
      } catch (err) {
        console.warn('Unable to clear chat state on reset', err);
      }
    }
  };

  const handleNavigate = (newView: AppView) => {
    if (newView === AppView.HOME) {
      if (analysis) {
        // Keep current session and return to results/chat instead of clearing state
        setView(AppView.RESULT);
      } else {
        resetApp();
      }
      return;
    }

    setPreviousView(view);
    setView(newView);
  };

  return (
    <div className="min-h-screen bg-earth-50 flex flex-col font-sans">
      <Header onNavigate={handleNavigate} />

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
                Show us your bug bite and describe the scene. We'll play detective and help you heal up!
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
          <AnalysisResult
            analysis={analysis}
            uploadedImage={uploadedImage}
            onReset={resetApp}
            messages={messages}
            onSendMessage={handleSendMessage}
            isChatLoading={isChatLoading}
            showSuggestions={showSuggestions}
            suggestions={remainingOptions}
            onSuggestionClick={handleSendMessage}
            onNavigateToGuide={() => handleNavigate(AppView.PREVENTION_GUIDE)}
          />
        )}

        {view === AppView.PREVENTION_GUIDE && (
          <PreventionGuide onBack={() => setView(previousView === AppView.RESULT ? AppView.RESULT : AppView.HOME)} />
        )}

        {view === AppView.SEASONAL_CALENDAR && (
          <SeasonalBugCalendar onBack={() => setView(previousView === AppView.RESULT ? AppView.RESULT : AppView.HOME)} />
        )}

        {view === AppView.BUG_EDUCATION && (
          <BugEducation onBack={() => setView(previousView === AppView.RESULT ? AppView.RESULT : AppView.HOME)} />
        )}

        {view === AppView.ABOUT && (
          <AboutPage onBack={() => setView(previousView === AppView.RESULT ? AppView.RESULT : AppView.HOME)} />
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
