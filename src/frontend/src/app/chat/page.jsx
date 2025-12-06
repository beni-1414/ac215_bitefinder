"use client";
import React, { useState } from "react";
import ChatInput from "../../components/chat/ChatInput";
import ChatMessage from "../../components/chat/ChatMessage";
import QuestionSuggestions from "../../components/chat/QuestionSuggestions";
import { evaluateBite, askRag } from "../../lib/DataService";
import { Card, CardHeader, CardContent } from "../../components/ui/card";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [remainingOptions, setRemainingOptions] = useState([]);
  const [lastUserMessage, setLastUserMessage] = useState("");
  const evalResRef = React.useRef(null);
  const [history, setHistory] = useState([]);
  const [hasPrediction, setHasPrediction] = useState(false);

  const allOptions = [
    "Relief & treatment",
    "Prevention tips",
    "About this insect",
  ];

  const addMessage = (role, text, image = null) => {
    setMessages((prev) => [...prev, { role, text, image }]);
  };


  // Utility function to convert image file to base 64 string
  async function imageToBase64(file) {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }


  // Advice string extraction helper (use everywhere you add RAG assistant messages)
  function extractAdvice(llm) {
    if (!llm) return "No advice was returned.";
    if (typeof llm === "string") return llm;
    if (Array.isArray(llm)) return llm.join('\n\n');

    // Prefer dedicated fields
    if (Array.isArray(llm.prevention_tips)) return llm.prevention_tips.join('\n\n');
    if (llm.prevention_tips) return llm.prevention_tips;
    if (Array.isArray(llm.treatment_for_mosquito_bites)) return llm.treatment_for_mosquito_bites.join('\n\n');
    if (llm.treatment_for_mosquito_bites) return llm.treatment_for_mosquito_bites;
    if (Array.isArray(llm.treatment_advice)) return llm.treatment_advice.join('\n\n');
    if (llm.treatment_advice) return llm.treatment_advice;
    if (llm.answer) return llm.answer;

    // Robustly extract info fields (recursively if needed)
    if (typeof llm.insect === "object" && llm.insect !== null) {
      // Join details from inner object
      return Object.values(llm.insect)
        .filter(v => typeof v === "string" && v.trim())
        .join('\n\n');
    }
    // If llm is a flat object with info keys:
    const infoStrings = Object.values(llm)
      .filter(v => typeof v === "string" && v.trim())
      .map(v => v.trim());
    if (infoStrings.length) return infoStrings.join('\n\n');

    return "No advice was returned.";
  }


  // -------------------------------
  // FOLLOW-UP QUESTION HANDLER
  // -------------------------------
  const handleQuestionSelect = async (question) => {
    if (!evalResRef.current) return;

    setShowSuggestions(false);
    setLoading(true);
    addMessage("user", question);

    const nextOptions = remainingOptions.filter((o) => o !== question);

    try {
      const ragRes = await askRag({
        question,
        symptoms: lastUserMessage,
        bug_class: evalResRef.current.prediction || "unknown",
        conf: evalResRef.current.confidence || 0.0,
      });

      const answer = extractAdvice(ragRes.llm);
      addMessage("assistant", answer);

      if (nextOptions.length > 0) {
        setRemainingOptions(nextOptions);
        setTimeout(() => setShowSuggestions(true), 400);
      }
    } catch (err) {
      console.error(err);
      addMessage("assistant", "Something went wrong while contacting the server.");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------
  // MAIN SEND HANDLER
  // ----------------------
  const handleSend = async ({ message, image }) => {
    setLoading(true);

    let imageURL = null;
  let image_base64 = null;

  if (image) {
    try {
      image_base64 = await imageToBase64(image);
      const blob = await (await fetch(`data:${image.type};base64,${image_base64}`)).blob();
      imageURL = URL.createObjectURL(blob);
    } catch (err) {
      console.error("Error reading image:", err);
      addMessage("assistant", "Failed to process the image.");
    }
  }

  addMessage("user", message || "[uploaded image]", imageURL);

    if (hasPrediction) {
      try {
        const relevanceCheck = await evaluateBite({
          user_text: message,
          image_gcs_uri: null,
          image_base64: image_base64,
          first_call: false,
          history: [],
        });

        if (relevanceCheck.eval?.courtesy) {
          addMessage("assistant", "You're welcome. Let me know if you have other questions about the bite.");
          setLoading(false);
          return;
        }

        if (!relevanceCheck.eval?.question_relevant) {
          const errorMsg = relevanceCheck.eval?.improve_message || "I can only answer questions about insect bites, symptoms, prevention, or treatment.";
          addMessage("assistant", errorMsg);
          setLoading(false);
          return;
        }

        // QUESTION IS RELEVANT → CALL RAG
        const ragRes = await askRag({
          question: message,
          symptoms: lastUserMessage,
          bug_class: evalResRef.current?.prediction || "unknown",
          conf: evalResRef.current?.confidence || 0.0,
        });

        const answer = extractAdvice(ragRes.llm);
        addMessage("assistant", answer);

      } catch (err) {
        console.error(err);
        addMessage("assistant", "Something went wrong while contacting the server.");
      } finally {
        setLoading(false);
      }
      return; // DO NOT CALL /evaluate ANYMORE
    }

    // FIRST MESSAGE → CALL /evaluate
    if (!hasPrediction) {
      setLastUserMessage(message || "");
    }

    setHistory((prev) => [...prev, message]);

    try {
      const evalResult = await evaluateBite({
        user_text: message,
        image_gcs_uri: null,
        image_base64: image_base64,
        first_call: true,
        history: history,
      });

      if (evalResult.needs_fix) {
        addMessage("assistant", evalResult.error || "Please provide more details.");
        setLoading(false);
        return;
      }

      const passed = evalResult.eval;
      evalResRef.current = passed;

      const pred = passed.prediction || "unknown";
      const conf = typeof passed.confidence === "number" ? passed.confidence : 0;

      addMessage(
        "assistant",
        `According to our AI engine, this appears to be a ${pred.replaceAll('_', ' ').slice(0, -1)} bite with ${Math.round(conf * 100)}% confidence.`
      );

      setRemainingOptions(allOptions);
      setShowSuggestions(true);
      setHasPrediction(true);
    } catch (err) {
      console.error(err);
      addMessage("assistant", "Oops, something went wrong while contacting the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-background px-4">
      <Card className="w-full max-w-2xl shadow-md border">
        <CardHeader>
          <h1 className="text-xl font-semibold text-center">BiteFinder Assistant</h1>
          <p className="text-sm text-muted-foreground text-center">
            Upload a photo and describe what happened.
            <br />
            I'll help figure out what might have caused the bite and how to treat it.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col justify-between h-[70vh]">
          <div className="flex-1 overflow-y-auto mb-4">
            {messages.map((m, i) => (
              <ChatMessage key={i} role={m.role} text={m.text} image={m.image} />
            ))}
            {loading && (
              <ChatMessage
                role="assistant"
                text="Analyzing your input, please wait..."
              />
            )}
            {showSuggestions && (
              <QuestionSuggestions
                options={remainingOptions}
                onSelect={handleQuestionSelect}
              />
            )}
          </div>
          <ChatInput
            onSend={handleSend}
            placeholderText={
              showSuggestions
                ? "Type a follow-up question..."
                : "Describe your symptoms and where you were bitten..."
            }
          />
        </CardContent>
      </Card>
    </div>
  );
}
