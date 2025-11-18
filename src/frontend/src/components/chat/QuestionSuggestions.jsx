"use client"
import React from "react"
import { Button } from "@/components/ui/button"

export default function QuestionSuggestions({ options, onSelect }) {
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {options.map((q, i) => (
        <Button
          key={i}
          variant="outline"
          className="text-sm rounded-full border-gray-500"
          onClick={() => onSelect(q)}
        >
          {q}
        </Button>
      ))}
    </div>
  )
}
