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


  const allOptions = [
    "Relief & treatment",
    "Prevention tips",
    "About this insect",
  ];

  const addMessage = (role, text, image = null) => {
    setMessages((prev) => [...prev, { role, text, image }]);
  };

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

      const answer =
        ragRes.llm?.answer ||
        ragRes.llm ||
        "I couldn't get advice for this bite. Try again later.";

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

  // -------------------------------
  // MAIN SEND HANDLER
  // -------------------------------
  const handleSend = async ({ message, image }) => {
    setLoading(true);

    setLastUserMessage(message || "");

    const imageURL = image ? URL.createObjectURL(image) : null;
    addMessage("user", message || "[uploaded image]", imageURL);

    setHistory((prev) => [...prev, message]);

    // ---- CALL /evaluate ----
    try {
      const evalResult = await evaluateBite({
        user_text: message,
        image_gcs_uri: null,
        first_call: history.length === 0,
        history: history,
      });

      // CASE 1: Validation fails
      if (evalResult.needs_fix) {
        addMessage("assistant", evalResult.error || "Please provide more details.");
        setLoading(false);
        return;
      }

      // CASE 2: Validation passes
      const passed = evalResult.eval;
      evalResRef.current = passed;

      const pred = passed.prediction || "unknown";
      const conf = typeof passed.confidence === "number" ? passed.confidence : 0;

      addMessage(
        "assistant",
        `According to our AI engine, this appears to be a ${pred} bite (confidence: ${conf.toFixed(
          2
        )}).`
      );

      setRemainingOptions(allOptions);
      setShowSuggestions(true);
    } catch (err) {
      console.error(err);
      addMessage(
        "assistant",
        "Oops, something went wrong while contacting the server."
      );
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------
  // RENDER
  // -------------------------------
  return (
    <div className="flex justify-center items-center min-h-screen bg-background px-4">
      <Card className="w-full max-w-2xl shadow-md border">
        <CardHeader>
          <h1 className="text-xl font-semibold text-center">BiteFinder Assistant</h1>
          <p className="text-sm text-muted-foreground text-center">
            Upload a photo and describe what happened.
            <br />
            I’ll help figure out what might have caused the bite and how to treat it.
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
