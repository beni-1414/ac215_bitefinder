PROMPT_TEMPLATE = """
You are labeling an image dataset of insect bites for a vision-language model.

Given:
- The known insect type: "{bite_type}"
- The image below shows the affected area.

Your task:
1. Fill the JSON fields following the descriptions below, using only information observable in the image.
2. Leave any unobservable fields as "UNKNOWN".
3. Output only valid JSON following this schema:

{{
  "body_location": string, # If impossible to determine, use UNKNOWN
  "number_of_bites": string, # one, two, three, four, five, more than 5
  "appearance": string, # Short description of the bite appearance
  "swelling": string, # none, mild, moderate, severe
  "color": string,
  "pattern": string, # single, clustered, linear, etc.
  "time_since_bite": string, # Guess in hours
}}
"""