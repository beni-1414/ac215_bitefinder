def build_prompt(batch, common_symptoms, selected_rare, category):
    return f"""
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