# Synthetic Label Generation

## Files Overview
### input/symptoms.json
Extraction of subjective sensory symptoms for insect and arachnid bites from Cleveland Clinic website. Symptoms are categorized by frequency (common vs. rare/allergic reactions) and structured in JSON format with separate lists for each bite type.

### input/locations.json
Generated plausible human-reported bite locations for 7 arthropod types. Creates 30 unique descriptions per type, mixing indoor/outdoor environments based on ecological knowledge and focusing on what people would reasonably remember when reporting bite incidents.

### label_generation.py
For every location, it generates 4 paraphrased versions including symptoms, vocab variability, bad answers, inconcrete answers, etc. to mimic real-world user reports.

## Usage
### Local
Running it from Windows (Git Bash) gives issues with mounting volumes. Use WSL or a Linux/Mac system.
```bash
sh src/synthetic_label_generation/docker-shell-local.sh
```
### VM
Ensure the VM is authenticated with the proper service account.
```bash
sh src/synthetic_label_generation/docker-shell-vm.sh
```

Once inside the container, run:
```bash
python3 src/synthetic_label_generation/label_generation.py
```

You will find the output both in the container and in the GCP bucket ``data/text/training/texts_{timestamp}.json``

## Versions
### v1.0
Only contains info regarding symptoms and location. No mention of body part nor number of bites (one, many...) which would both be common in real world scenario. 