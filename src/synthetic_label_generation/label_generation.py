import os
import time
import random
from prompt import build_prompt

from utils import (
    init_openai_client,
    load_json_file,
    generate_batch_labels,
    save_json,
    upload_to_gcs,
    timestamp_suffix
)


def main(test_mode: bool = False):
    # Initialize OpenAI client
    client = init_openai_client()

    # Load input data
    locations_data = load_json_file("input/locations.json")
    symptoms_data = load_json_file("input/symptoms.json")

    # Prepare results dict with the same 7 keys
    results = {category: [] for category in locations_data.keys()}

    # Loop through each category
    for category, locations in locations_data.items():
        common_symptoms = symptoms_data[category]["common"]
        rare_symptoms = symptoms_data[category]["rare"]

        # Process locations in batches of up to 5
        for i in range(0, len(locations), 5):
            batch = locations[i:i+5]
            selected_rare = random.choice(rare_symptoms)
            prompt = build_prompt(batch, common_symptoms, selected_rare, category)

            if test_mode:
                if i >= 5:
                    break # Only do one batch in test mode

            try:
                batch_json = generate_batch_labels(
                    client=client,
                    model="gpt-4o-mini",
                    prompt=prompt
                )

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

            # Sleep to avoid rate limits
            time.sleep(1.5)

    # Save final results with timestamp
    suffix = timestamp_suffix()
    synthetic_filename = f"synthetic_bite_labels_{suffix}.json"
    aggregate_filename = f"synthetic_bite_labels_aggregate_{suffix}.json"

    output_file = save_json(results, synthetic_filename)

    # Create aggregate file
    aggregate = {}
    for category, entries in results.items():
        all_sentences = []
        for entry in entries:
            all_sentences.extend(entry["sentences"])
        aggregate[category] = all_sentences

    aggregate_file = save_json(aggregate, aggregate_filename)

    print(f"\n✅ All synthetic labels generated and saved to:")
    print(f"   - {output_file}")
    print(f"   - {aggregate_file}")

    # Upload to GCS if bucket name is set
    gcp_bucket_name = os.getenv("GCP_BUCKET_NAME")
    save_path_gcp = os.getenv("GCP_PATH_SYNTHETIC_LABELS", "")
    if gcp_bucket_name:
        try:
            upload_to_gcs(gcp_bucket_name, output_file, save_path_gcp + synthetic_filename)
            upload_to_gcs(gcp_bucket_name, aggregate_file, save_path_gcp + aggregate_filename)
        except Exception as e:
            print(f"⚠️ Failed to upload to GCP: {e}")


if __name__ == "__main__":
    # Read the parameter test_mode from args if there is one
    test_mode = False
    args = os.sys.argv
    if len(args) > 1 and args[1].lower() == "test":
        test_mode = True
        print("⚠️ Running in TEST MODE - only one batch per category will be processed")
    main(test_mode=test_mode)
