# Audio Context Layer - Curio Next PoC

An end-to-end proof of concept for answering natural-language questions grounded in an audio clip.

## Approach

**Audio -> temporal segmentation -> audio feature extraction -> event/environment detection -> structured context -> question routing -> grounded answer**

The PoC uses a small, intentionally curated synthetic dataset so that every event has exact timestamps and ground-truth QA pairs.

The pipeline operates on 1-second temporal windows and produces a structured timeline containing detected events, timestamps, confidence scores, and the predicted environment. A question router then maps different question types to operations over this structured context.

Supported question types include:

- **Perceptual:** What sounds are present?
- **Environment:** What environment does this audio suggest?
- **Presence:** Is a particular sound present?
- **Counting:** How many times does an event occur?
- **Temporal:** What happens before/after an event?
- **Causal/reasoning:** Why does the audio appear to belong to a particular environment?

The project also includes a compact PyTorch log-mel CNN training baseline and training history/loss curve. The reported PoC evaluation uses the saved lightweight event and environment classifiers together with the structured context layer.

> **Important limitation:** This is a controlled proof of concept, not a general-purpose audio foundation model. The synthetic dataset is deliberately small and the model is trained only on the defined sound vocabulary. The reported quantitative results therefore demonstrate the implemented pipeline on the controlled benchmark rather than general real-world audio understanding.

## Project Structure

```text
curio_audio_context/
|-- data/
|   |-- audio/
|   |-- train.json
|   |-- val.json
|   |-- test.json
|   `-- dataset_meta.json
|-- models/
|-- results/
|-- src/
|   |-- generate_dataset.py
|   |-- audio_utils.py
|   |-- model.py
|   |-- train.py
|   |-- context_layer.py
|   |-- evaluate.py
|   |-- demo.py
|   `-- api.py
|-- docs/
|   |-- TECHNICAL_REPORT.md
|   `-- TECHNICAL_REPORT.pdf
|-- requirements.txt
`-- README.md