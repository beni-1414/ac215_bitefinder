BENI brainstorming (no chatgpt):

Global warming is expanding bug habitats into cooler regions (e.g., central Europe, northern 
Spain, New England), causing more unfamiliar bites (e.g., this summer’s tick plagues, invasive 
wasps, new mosquitos). Seeing a dermatologist is inconvenient, appointments are scarce and 
delayed, and most of the times you are in a remote area with almost no access to healthcare. Also, depending on the bite the care provider might not even know (I can explain my bed bug experience this summer), so our target users are hikers and campers in remote areas, as well as general population that might encounter an unknown bite and want to get a recomendation from a reputable source rather than ChatGPT, which sometimes works well but some times just invents the answer or even plainly refuses to answer.

Develop an bug bite classification web-app, where a user can take a photo of their skin on 
their phone and type their symptoms, and the app will predict what bug caused the bite and 
provide recommendations on which products to use, prognostic or whether to see a doctor, information about the insect...

As future steps, we could make the webapp available as an MCP for ChatGPT, Claude, etc, so the app reach benefits from the default most young people do to this chatbots for everything. The money would come from the product recomendations for treating the bites, like sponsored ads. There would be a check that all medicines are actually useful before accepting them into the RAG system (ethical consideration).

THe solution is more transparent than ChatGPT in the sense that it uses a multimodal classifier trained specifically on this task, with clear evaluation and filtered reputable sources for medical information regarding prognosis. *It will also be equipped with guardrails to send people to the hospital in case of a suspicious tick bite or an infected one (extra classifier head)* - I @Beni tried to find data for this and did not find it. We could try to scrap it only for severe tick bites, and get 10-20 images for a small classifier, but it is not a priority.

As value, we can show numbers regarding how hard it is to get to a PCP in the US, how much money the bug bite pharma moves, and some numbers regarding mosquito, wasp, tick, etc population shifts due to climate change.

Regarding the technical scalability, we shall explain the diagram, mention the technologies chosen for each part, and how each one relates to the added value of our tool. Save 1 slide for this, we will fill it up.

At the moment, we have the rag and the classification head working, as well as the preprocessing steps. We are working on a quality evaluation layer, the overall agentic coordination of the system, how to treat ambiguous images and the implementation of a "biased" RAG system that shows sponsored products first. If time allows, we will make the tool available as an MCP as well as a web-app.

Chatgpt Outline + beni post-processing

### Slide 1 — Title & One-Liner (0:30)

Title: Snap → Know the Bite → Take the Right Next Step
Subtitle: Task-specific, transparent AI for insect bite identification

Say:
“Global warming is moving insect habitats into new regions. When people get an unfamiliar bite on a trail, on a trip or at home they need quick, reliable guidance. Our app lets users snap a photo, add symptoms and environment description, and receive an evidence-based bite classification with clear, sourced advice.”

Visual: Product hero mock (phone with camera → result screen). One killer stat (e.g., tick incidence trend) in small text.

### Slide 2 — Problem & Users (0:45)

Title: The Pain Today
Bullets (2–3 max):
* Hard to access dermatology quickly (*add some statistic of system burden*)
* LLMs like ChatGPT are good, but non-transparent and tend to reject answering medical questions (*quickly pop up example*)
* Most bites resolve quickly with good care and OTC medicines, you just need proper guidance.

Say:
"need to decide"

Visual: Simple journey: Bite → Confusion → Our app → Actionable info.

### Slide 3 — Solution & Demo Flow (0:50)

Title: What It Does
Panels: (1) Photo capture, (2) Model prediction + confidence, (3) Sourced guidance

Say:
The aim of this slide should be a walkthrough of what the user sees and does, no technical details yet.

“Take a photo, type symptoms and location information. Our model classifies likely culprit bites and shows guidance from vetted sources—what to do now, what to watch for, product options, preventive care for the future... All delivered via a familiar chatbot interface, so you ask and we answer. If confidence is low, we say so and encourage caution—no guesswork.”


### Slide 4 — Why We’re Different (0:40)

Title: Task-Specific, Transparent, Cautious
Bullets:

* Task-specific multimodal classifier (images + symptoms), not a general LLM

* Transparent RAG: only vetted medical sources, citations visible

* Safety first: image-quality check + uncertainty-based abstain (triage classifier is being studied for better "you should see a doctor" recommendations, specially for tick bites).

### Slide 5 — Tech & MLOps Architecture (1:05)

Title: Scalable, Monitorable, On-cloud

Visual: Zoe's plot

Say:
Explain the plot, mention the choosen technologies, and also mention that we do not store patient data, only their locations and predicted bug bites for epidemiological studies on insect populations (CDC might be interested, return to society and extra revenue).


### Slide 6 — Current Performance & Safety (0:35)

Title: Early Results & Safety Posture
Tiny table (placeholders—fill with your numbers):

Some accuracy metrics from the training + 1 number for inference time + 1 number for cost per inference

Bullets:

* Data quality gate to ensure model performance
* Currently running experiments to improve metrics and tackle edge cases.
* Implementing proper handling of uncertain cases, no guesswork, no hallucinations.

### Slide 7 — Market, GTM & Business Model (0:35)

Title: Who We Reach & How We Fund It
Bullets:

* Beachhead: outdoor retail & pharmacies (QR at point-of-sale, trailheads), ads on instagram and trail related apps (Wikiloc, etc)
* Monetization v1: Sponsored/affiliate products with ethical whitelist + clear labels
* Future reach: wrap in an MCP that can be used in main LLM providers, using their interface to pass our predictions and recomendations.

Say:
Low-frequency usage means we keep it free for consumers and fund via clearly labeled, vetted product placements. Distribution comes from partners where bites are top-of-mind.

### Slide 8 — Roadmap, Risks & The Ask (0:30)

Title: What’s Next

* Ship: quality gate, confidence-aware abstain, RAG with vetted sources
* MLOps: CI/CD, MLflow pipeline build...
* Validation: more training on the cloud, prospective sample, update metrics

Future work (out of current scope):

* Triage head (doctor-now vs. self-care): challenge getting appropriate data
* On-device/quantized model for online limited use - requires mobil app dev (maybe better not to say it)
* Some idea