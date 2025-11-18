"use client"
import React, { useState } from "react"
import ChatInput from "../../components/chat/ChatInput"
import ChatMessage from "../../components/chat/ChatMessage"
import QuestionSuggestions from "../../components/chat/QuestionSuggestions"
import { evaluateBite, askRag } from "../../lib/DataService"
import { Card, CardHeader, CardContent } from "../../components/ui/card"

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [remainingOptions, setRemainingOptions] = useState([])
  const [lastEval, setLastEval] = useState(null)
  const [lastUserMessage, setLastUserMessage] = useState("");
  const evalResRef = React.useRef(null);


  const allOptions = [
    "Relief & treatment",
    "Prevention tips",
    "About this insect",
  ]

  const addMessage = (role, text, image = null) => {
    setMessages((prev) => [...prev, { role, text, image }])
  }

  const handleQuestionSelect = async (question) => {
    if (!lastEval) return
    setShowSuggestions(false)
    setLoading(true)
    addMessage("user", question)

    // filter out the question that was just chosen
    const nextOptions = remainingOptions.filter((q) => q !== question)

    try {
      const ragRes = await askRag({
        question,
        symptoms: lastUserMessage,
        bug_class: evalResRef.current?.prediction || "unknown",
        conf: evalResRef.current?.confidence || 0.0,
      })

      const response =
        ragRes.llm?.answer ||
        ragRes.llm ||
        "I couldn’t find advice for this bite. Try again later."

      addMessage("assistant", response)

      // ✅ show remaining questions as follow-ups
      if (nextOptions.length > 0) {
        setRemainingOptions(nextOptions)
        setTimeout(() => setShowSuggestions(true), 400) // small delay for flow
      }
    } catch (err) {
      console.error(err)
      addMessage("assistant", "Something went wrong while contacting the server.")
    } finally {
      setLoading(false)
    }
  }

  const handleSend = async ({ message, image }) => {
    setLoading(true)

    // ⭐ ADD THIS — store the user’s message so RAG follow-ups know the symptoms
    setLastUserMessage(message || "")

    let imageURL = image ? URL.createObjectURL(image) : null
    addMessage("user", message || "[uploaded image]", imageURL)

    try {
      const evalRes = await evaluateBite({
        user_text: message,
        image_gcs_uri: null,
      })

      if (!evalRes || evalRes.error) {
        addMessage("assistant", evalRes?.error || "Please try again.")
        setLoading(false)
        return
      }

      // ⭐ ADD THIS — store eval results for follow-up questions
      evalResRef.current = evalRes

      // You already had this, keep it
      setLastEval({ ...evalRes, symptoms: message })

      addMessage(
        "assistant",
        `According to our AI engine, this appears to be a ${
          evalRes.prediction || "unknown"
        } bite (confidence: ${(evalRes.confidence || 0).toFixed(2)}).`
      )

      // show the suggestion buttons
      setRemainingOptions(allOptions)
      setShowSuggestions(true)
    } catch (err) {
      console.error(err)
      addMessage("assistant", "Something went wrong while contacting the server.")
    } finally {
      setLoading(false)
    }
  }


  return (
    <div className="flex justify-center items-center min-h-screen bg-background px-4">
      <Card className="w-full max-w-2xl shadow-md border">
        <CardHeader>
          <h1 className="text-xl font-semibold text-center">BiteFinder Assistant</h1>
          <p className="text-sm text-muted-foreground text-center">
            Upload a photo and describe what happened.
            <br />I’ll help figure out what might have caused the bite and how to treat it.
          </p>
        </CardHeader>

        <CardContent className="flex flex-col justify-between h-[70vh]">
          <div className="flex-1 overflow-y-auto mb-4">
            {messages.map((m, i) => (
              <ChatMessage key={i} role={m.role} text={m.text} image={m.image} />
            ))}

            {loading && (
              <ChatMessage role="assistant" text="Analyzing your input, please wait..." />
            )}

            {showSuggestions && (
              <QuestionSuggestions options={remainingOptions} onSelect={handleQuestionSelect} />
            )}
          </div>

          <ChatInput
            onSend={handleSend}
            placeholderText={
              showSuggestions
                ? "Type your own follow-up question..."
                : "Describe your symptoms and where you were bitten..."
            }
          />
        </CardContent>
      </Card>
    </div>
  )
}
