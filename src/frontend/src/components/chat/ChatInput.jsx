"use client"
import React, { useState } from "react"

export default function ChatInput({ onSend, placeholderText = "Type your message..." }) {
  const [message, setMessage] = useState("")
  const [image, setImage] = useState(null)

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (message.trim() !== "" || image) {
        onSend({ message, image })
        setMessage("")
        setImage(null)
      }
    }
  }

  const handleSubmit = () => {
    if (message.trim() !== "" || image) {
      onSend({ message, image })
      setMessage("")
      setImage(null)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {/* TEXTAREA */}
      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholderText}
        className="w-full border rounded-md p-2 text-sm resize-none
                   focus:outline-none focus:ring-2 focus:ring-blue-400
                   transition-all duration-300 bg-background"
        rows={3}
      />

      {/* IMAGE + SEND BUTTON */}
      <div className="flex items-center gap-2">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files?.[0] || null)}
          className="text-sm text-muted-foreground"
        />

        <button
          onClick={handleSubmit}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm"
        >
          Send
        </button>
      </div>
    </div>
  )
}
