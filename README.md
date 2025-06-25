# Encoder Model Stealing Attack 🔓

This repository implements a model-stealing attack on a remote image encoder exposed via an API.  
The attack follows a query-based workflow: collect embeddings from the victim model, train a surrogate (stolen) model to mimic it, and export the result for evaluation.

## Workflow Overview

### Session Setup
- Retrieve your seed and port using the provided API token.

### Dataset Preparation
- Load the base dataset (`ModelStealingPub.pt`).
- Apply one random augmentation per image to increase diversity and bypass server-side defenses.

### Embedding Query
- Query the target encoder in batches of 1000.
- Base64-encode images and send to the API.
- Store 1024-dim embeddings in `embeddings.pickle`.

### Surrogate Model Training
- Use a modified ResNet-18 to predict embeddings from images.
- Train using MSE loss to match target outputs.

### Export & Submit
- Convert the trained model to ONNX.
- Submit the file to the evaluation server with your token and seed.

| File                        | Description                             |
| --------------------------- | --------------------------------------- |
| `model_details.py`           | Launches session, gets seed & port      |
| `stealing_query.py`          | Queries target API and saves embeddings |
| `train_encoder.py`           | Trains the surrogate model              |
| `export_onnx.py`             | Converts PyTorch model to ONNX          |
| `submit.py`                  | Submits ONNX model for evaluation       |
| `ModelStealingPub.pt`        | Provided image dataset                  |
| `ModelStealingPub_augmented.pt` | Augmented image dataset               |
| `embeddings.pickle`          | Queried embeddings (generated)          |

## Notes
- Wait 60 seconds between queries to avoid rate limits.
- Data augmentations help avoid B4B defense (active anti-theft).
- Code is reproducible: behavior controlled via seed from server.

## References
Key ideas from:
- Han et al., *"On Defenses Against Model Stealing via B4B"*, NeurIPS 2023
- Chen et al., *"Encoder Stealing via Augmentations"*
- Sha et al., *"Copycat Networks"*

