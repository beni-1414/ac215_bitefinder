// frontend/src/components/chat/ChatMessage.jsx
"use client"
import React from "react"
import { Card, CardContent } from "@/components/ui/card"

export default function ChatMessage({ role, text, image }) {
  const isUser = role === "user"
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
      <Card
        className={`max-w-[80%] ${
          isUser ? "bg-blue-100 dark:bg-blue-900" : "bg-gray-100 dark:bg-gray-800"
        }`}
      >
        <CardContent className="p-3 text-sm whitespace-pre-wrap">
          {image && (
            <img
              src={image}
              alt="Uploaded"
              className="mb-2 rounded-lg max-h-48 object-cover"
            />
          )}
          {text}
        </CardContent>
      </Card>
    </div>
  )
}
