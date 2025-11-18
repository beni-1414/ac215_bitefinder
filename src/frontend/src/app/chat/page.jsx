"use client"
import React, { useState } from "react"
import ChatInput from "../../components/chat/ChatInput"
import ChatMessage from "../../components/chat/ChatMessage"
import { evaluateBite, askRag } from "../../lib/DataService"
import { Card, CardHeader, CardContent } from "../../components/ui/card"

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  // helper to add messages (supports optional image)
  const addMessage = (role, text, image = null) => {
    setMessages((prev) => [...prev, { role, text, image }])
  }

  const handleSend = async ({ message, image }) => {
    setLoading(true)

    // 1. create a local image preview (doesn't upload yet)
    let imageURL = null
    if (image) {
      imageURL = URL.createObjectURL(image)
    }

    // 2. immediately show user's message + image in chat
    addMessage("user", message || "[uploaded image]", imageURL)

    try {
      // 3. call orchestrator evaluate endpoint
      const evalRes = await evaluateBite({
        user_text: message,
        image_gcs_uri: null, // VLM skipped for now
      })

      // 4. handle evaluation errors
      if (!evalRes || evalRes.error) {
        addMessage(
          "assistant",
          evalRes?.image_issue ||
            evalRes?.text_issue ||
            evalRes?.error ||
            "Please try again with a clearer photo or more detail."
        )
        setLoading(false)
        return
      }

      // 5. show the classification result first
      addMessage(
        "assistant",
        evalRes.message ||
          `Detected possible ${evalRes.prediction || "unknown"} bite.`
      )


      // 6. call RAG model for treatment advice
      const ragRes = await askRag({
        question: "treatment and prevention advice",
        symptoms: message,
        bug_class: evalRes.prediction || "unknown",
        conf: evalRes.confidence || 0.0,
      })

      const response =
        ragRes.llm?.answer ||
        ragRes.llm ||
        "I couldn’t find advice for this bite. Try again later."

      // 7. show RAG model’s advice
      addMessage("assistant", response)
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
          <h1 className="text-xl font-semibold text-center">
            BiteFinder Assistant
          </h1>
          <p className="text-sm text-muted-foreground text-center">
            Upload a photo and describe what happened.
            <br />
            I’ll help figure out what might have caused the bite and how to treat it.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col justify-between h-[70vh]">
          <div className="flex-1 overflow-y-auto mb-4">
            {messages.map((m, i) => (
              <ChatMessage
                key={i}
                role={m.role}
                text={m.text}
                image={m.image}
              />
            ))}
            {loading && (
              <ChatMessage
                role="assistant"
                text="Analyzing your input, please wait..."
              />
            )}
          </div>
          <ChatInput onSend={handleSend} />
        </CardContent>
      </Card>
    </div>
  )
}
