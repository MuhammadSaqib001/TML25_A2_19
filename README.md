# Encoder Model Stealing Attack 🔓

This repository implements a model-stealing attack on a remote image encoder exposed via an API.  
The attack follows a query-based workflow: collect embeddings from the victim model, train a surrogate (stolen) model to mimic it, and export the result for evaluation.

## 🔧 Workflow Overview

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
