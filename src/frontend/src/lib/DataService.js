// src/lib/DataService.js

export async function evaluateBite(data) {
  const res = await fetch(`/api/v1/orchestrator/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_gcs_uri: data.image_gcs_uri || null,
      image_base64: data.image_base64 || null,
      user_text: data.user_text || null,
      overwrite_validation: false, // <-- DO NOT OVERRIDE VALIDATION
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

  // IMPORTANT FIX:
  // If validation failed -> STOP HERE.
  if (result.ok === false) {
    return {
      needs_fix: true,
      error: result.text_issue || result.image_issue,
      eval: result
    };
  }

  // Otherwise evaluation passed.
  // FIX - unwrap eval so frontend receives a flat object
  return {
    needs_fix: false,
    eval: result.eval ? result.eval : result // Only pass the inner eval if present
  };
}

// Add this to fix your error!
export async function askRag(data) {
  const res = await fetch(`/api/v1/orchestrator/rag-agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server returned ${res.status}: ${text}`);
  }
  return res.json();
}
