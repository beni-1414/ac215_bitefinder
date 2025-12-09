// src/services/dataService.ts

export interface EvaluateRequest {
  image_gcs_uri?: string | null;
  image_base64?: string | null;
  user_text?: string | null;
  overwrite_validation: boolean;
  first_call: boolean;
  history: string[];
  return_combined_text: boolean;
  debug: boolean;
}

export interface EvaluateResponse {
  needs_fix: boolean;
  error?: string;
  eval?: {
    prediction?: string;
    confidence?: number;
    courtesy?: boolean;
    question_relevant?: boolean;
    improve_message?: string;
    [key: string]: any;
  };
}

export interface RagRequest {
  question: string;
  symptoms: string;
  bug_class: string;
  conf: number;
  session_id?: string;
}

export interface RagResponse {
  llm?: any;
  session_id?: string;
  [key: string]: any;
}

const RAG_SESSION_STORAGE_KEY = "bitefinder_rag_session_id";

const getStoredRagSessionId = (): string | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(RAG_SESSION_STORAGE_KEY);
  } catch (err) {
    console.warn("Unable to read RAG session ID from storage", err);
    return null;
  }
};

const persistRagSessionId = (sessionId: string) => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(RAG_SESSION_STORAGE_KEY, sessionId);
  } catch (err) {
    console.warn("Unable to persist RAG session ID", err);
  }
};

export const clearRagSession = () => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(RAG_SESSION_STORAGE_KEY);
  } catch (err) {
    console.warn("Unable to clear RAG session ID", err);
  }
};

/**
 * Calls the backend /evaluate endpoint
 */
export async function evaluateBite(data: {
  image_gcs_uri?: string | null;
  image_base64?: string | null;
  user_text?: string | null;
  first_call: boolean;
  history?: string[];
}): Promise<EvaluateResponse> {
  const res = await fetch(`/v1/orchestrator/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_gcs_uri: data.image_gcs_uri || null,
      image_base64: data.image_base64 || null,
      user_text: data.user_text || null,
      overwrite_validation: false,
      first_call: data.first_call,
      history: data.history || [],
      return_combined_text: true,
      debug: false,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Server returned ${res.status}: ${errText}`);
  }

  const result = await res.json();

  // If validation failed -> STOP HERE
  if (result.ok === false) {
    const errorMsg = result.text_issue || result.image_issue || "Please provide more details.";
    console.log("Validation failed, error:", errorMsg);
    return {
      needs_fix: true,
      error: errorMsg,
      eval: result
    };
  }

  // Otherwise evaluation passed
  console.log("Validation passed, returning eval");

  // FIX: For follow-up calls, the backend returns {ok: true, eval: {...}}
  // We need to extract the inner 'eval' field to avoid double-nesting
  return {
    needs_fix: false,
    eval: result.eval || result  // ← Use result.eval if it exists, otherwise use result
  };
}

/**
 * Calls the backend /rag endpoint for follow-up questions
 */
export async function askRag(data: RagRequest): Promise<RagResponse> {
  console.log('🟢 Calling RAG with:', data);  // ← Added debugging

  const payload: RagRequest = { ...data };
  const storedSessionId = getStoredRagSessionId();
  if (storedSessionId) {
    payload.session_id = storedSessionId;
  }

  const res = await fetch(`/v1/orchestrator/rag-agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  console.log('🟢 RAG response status:', res.status);  // ← Added debugging

  if (!res.ok) {
    const text = await res.text();
    console.error('❌ RAG error response:', text);  // ← Added debugging
    throw new Error(`Server returned ${res.status}: ${text}`);
  }

  const result = await res.json();
  console.log('🟢 RAG response:', result);  // ← Added debugging

  if (result?.session_id) {
    persistRagSessionId(result.session_id);
  }

  return result;
}

/**
 * Utility: Extract advice text from LLM response
 */
export function extractAdvice(llm: any): string {
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

  // Robustly extract info fields
  if (typeof llm.insect === "object" && llm.insect !== null) {
    return Object.values(llm.insect)
      .filter(v => typeof v === "string" && v.trim())
      .join('\n\n');
  }

  const infoStrings = Object.values(llm)
    .filter(v => typeof v === "string" && v.trim())
    .map(v => (v as string).trim());

  if (infoStrings.length) return infoStrings.join('\n\n');

  return "No advice was returned.";
}
