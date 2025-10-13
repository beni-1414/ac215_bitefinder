This side pipeline was my initial attempt at using GPT to create the fake labels. I then realized that most of the info I had envisioned (appearance, colour...) would be in theory captured by the vision model, and since the time_from_bite was not performing good I discarded it.

Might be useful if we want to include the body part or the amount of bites in the textual prompt, to avoid the text and image contradicting each other (itchy bumps -> image shows one bump).

NOT CONTAINERIZED.