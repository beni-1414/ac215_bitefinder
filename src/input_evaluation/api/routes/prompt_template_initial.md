You are evaluating a user's first message about a bug bite.

USER MESSAGE:
{content}

INSTRUCTIONS:
1. Check if the message mentions symptoms (itchy, painful, red, swelling, bumps, rash, burning)
2. Check if the message mentions location/context (camping, forest, park, hiking, bed, outdoors)
3. Check if the message is about bug bites or insect bites

RULES:
- If message mentions bug bites/insects AND has symptoms: question_relevant = true
- If message is "hello", "hi", or other greetings: question_relevant = false
- If message has BOTH symptoms AND location: complete = true
- If message missing symptoms OR location: complete = false, improve_message = "For best results, tell us your symptoms and where you were when you got bit."

DANGER CHECK:
- If mentions "difficulty breathing" OR "throat swelling" OR "face swelling" OR "dizziness":
  Set high_danger = true, complete = false, question_relevant = true, improve_message = "Your description indicates potentially dangerous symptoms. Please seek immediate medical attention."

OUTPUT FORMAT (JSON only, no other text):
{"complete": true/false, "improve_message": "text or null", "combined_text": null, "high_danger": true/false, "question_relevant": true/false, "courtesy": false}

You MUST output ONLY a JSON object of the form:

{ "complete": <true/false>, "improve_message": <string or null>, "combined_text": null, "high_danger": <true/false>, "question_relevant": <true/false>, "courtesy": <true/false> }
