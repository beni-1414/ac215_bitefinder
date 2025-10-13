import json
import random
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
# Try to load from relative path first (for local development)
# Then try from environment variables (for containerized environment)
if os.path.exists('../../../.env'):
    load_dotenv('../../../.env')
else:
    load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in your env)
client = OpenAI(api_key=api_key)

# Load input data
with open("locations.json", "r") as f:
    locations_data = json.load(f)

with open("symptoms.json", "r") as f:
    symptoms_data = json.load(f)

# Prepare results dict with the same 7 keys
results = {category: [] for category in locations_data.keys()}

# Loop through each of the 7 categories
for category, locations in locations_data.items():
    common_symptoms = symptoms_data[category]["common"]
    rare_symptoms = symptoms_data[category]["rare"]

    # Process locations in batches of up to 5
    for i in range(0, len(locations), 5):
        # For testing, only process two batches per category
        if i >= 10:
            break
        batch = locations[i:i+5]

        # Randomly pick one rare symptom if needed
        selected_rare = random.choice(rare_symptoms)

        # Build the dynamic prompt
        prompt = f"""
You are generating synthetic patient-style descriptions of insect bites for training a vision language model.

Context:
- Locations: {', '.join(batch)}
- Common symptoms: {', '.join(common_symptoms)}
- Rare symptoms (optional): {selected_rare}
- The biting insect type corresponds to the category: {category}.

Instructions:
- For each location, create 4 short paraphrased sentences that sound like what a *patient might say* when describing their bite. They will be prompted to answer "Can you describe where you think you were bitten and what symptoms you are experiencing?"
- Never mention the insect type explicitly.
- Do not mention specific body parts as they are not provided in the metadata, just use the location context.
- Include uncertainty or lack of knowledge about the bite origin in some sentences.
- Sentences should sound natural and vary in tone and technical detail:
  - Some should use medical terms (e.g., "erythema" instead of "redness")
  - Most of them should be more casual (e.g., "bite got really itchy and swollen").
  - Some can be very brief (e.g., "Got bitten while hiking, itchy").
  - Some can be more elaborate (e.g., "I was camping in a forest and woke up with several itchy bumps, not sure what bit me").
- Always include common symptoms to some extent, no need to include all.
- Include the rare symptom naturally into *one* of the sentences.
- Return the result in JSON format with this structure:

{{
  "location_1": [
    "sentence1",
    "sentence2",
    "sentence3",
    "sentence4"
    ],
    "location_2": [
    "sentence1",
    "sentence2",
    "sentence3",
    "sentence4"
    ]...
}}

EXAMPLES:
- I was running through a dense forest, itchy bump.
- Lots of very itchy warm bumps, no idea where I got them. I am travelling and sleeping in a hostel.
- I think I got bitten while hiking in the mountains, bite is itchy.
- I do not know where I got bitten, just woke up with the bites.
etc.
"""
        # Send request
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                response_format={"type": "json_object"}
            )

            output_text = response.choices[0].message.content.strip()
            batch_json = json.loads(output_text)

            # Append to results
            for location, sentences in batch_json.items():
                results[category].append({
                    "location": location,
                    "sentences": sentences
                })

            print(f"✅ Generated for category '{category}', locations {batch}")

        except Exception as e:
            print(f"⚠️ Error for {category} / {batch}: {e}")
            continue

        # Sleep a bit to avoid rate limiting
        time.sleep(1.5)
        print(batch_json)

# Save final results
# Use output directory if it exists (for containerized environment)
output_dir = "output" if os.path.exists("output") else "."
output_file = os.path.join(output_dir, "synthetic_bite_labels.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All synthetic labels generated and saved to {output_file}")
