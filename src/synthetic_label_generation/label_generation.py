import json
import random
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from prompt import build_prompt
from google.cloud import storage
from google.cloud import secretmanager

def get_secret(secret_id: str):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

api_key = get_secret("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in your env)
client = OpenAI(api_key=api_key)

# Load input data from bucket
with open("input/locations.json", "r") as f:
    locations_data = json.load(f)

with open("input/symptoms.json", "r") as f:
    symptoms_data = json.load(f)

# Prepare results dict with the same 7 keys
results = {category: [] for category in locations_data.keys()}

# Loop through each of the 7 categories
for category, locations in locations_data.items():
    common_symptoms = symptoms_data[category]["common"]
    rare_symptoms = symptoms_data[category]["rare"]

    # Process locations in batches of up to 5
    for i in range(0, len(locations), 5):
        batch = locations[i:i+5]

        # Randomly pick one rare symptom if needed
        selected_rare = random.choice(rare_symptoms)

        # Build the dynamic prompt
        prompt = build_prompt(batch, common_symptoms, selected_rare, category)
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

# Save final results
# Use output directory if it exists (for containerized environment)
output_dir = "output" if os.path.exists("output") else "."
output_file = os.path.join(output_dir, "synthetic_bite_labels.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All synthetic labels generated and saved to {output_file}")

# Create an aggregate file with all sentences, structure {insect : [list of sentences]}
aggregate = {}
for category, entries in results.items():
    all_sentences = []
    for entry in entries:
        all_sentences.extend(entry["sentences"])
    aggregate[category] = all_sentences
aggregate_file = os.path.join(output_dir, "synthetic_bite_labels_aggregate.json")
with open(aggregate_file, "w") as f:
    json.dump(aggregate, f, indent=2)

# Upload to Google Cloud Storage if GCP_BUCKET_NAME is set
gcp_bucket_name = os.getenv("GCP_BUCKET_NAME")
if gcp_bucket_name:
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcp_bucket_name)

        blob = bucket.blob("synthetic_bite_labels.json")
        blob.upload_from_filename(output_file)

        # Upload aggregate file as well
        aggregate_blob = bucket.blob("synthetic_bite_labels_aggregate.json")
        aggregate_blob.upload_from_filename(aggregate_file)
        print(f"✅ Uploaded {output_file} to GCP bucket {gcp_bucket_name}")
    except Exception as e:
        print(f"⚠️ Failed to upload to GCP: {e}")