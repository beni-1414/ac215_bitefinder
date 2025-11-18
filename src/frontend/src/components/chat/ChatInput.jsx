// frontend/src/components/chat/ChatInput.jsx
"use client"
import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export default function ChatInput({ onSend }) {
  const [message, setMessage] = useState("")
  const [image, setImage] = useState(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!message && !image) return
    onSend({ message, image })
    setMessage("")
    setImage(null)
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-2 p-2 border-t border-gray-200 dark:border-gray-700"
    >
      <Textarea
        placeholder="Describe your symptoms and where you were bitten..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        className="resize-none"
        rows={3}
      />
      <div className="flex items-center justify-between">
        <Input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          className="max-w-xs"
        />
        <Button type="submit">Send</Button>
      </div>
    </form>
  )
}
