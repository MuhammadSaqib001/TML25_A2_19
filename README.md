# Encoder Model Stealing Attack 🔓

This repository implements a model-stealing attack on a remote image encoder exposed via an API.  
The attack follows a query-based workflow: collect embeddings from the victim model, train a surrogate (stolen) model to mimic it, and export the result for evaluation.

## Workflow Overview

### 1. Session Setup
- Retrieve your seed and port using the provided API token.

### 2. Dataset Preparation 
- Load the base dataset (`ModelStealingPub.pt`).
- Apply one random augmentation per image to increase diversity and bypass server-side defenses.
- Storing augmented samples as (`ModelStealingPub_augmented.pt`).

### 3. Embedding Query (Target Model)
- Query the target encoder using  (`ModelStealingPub.pt`) and (`ModelStealingPub_augmented.pt`) in batches of 1000.
- Base64-encode images and send to the API.
- Store 1024-dim embeddings in `embeddings.pickle` for all the input images.

### 4. Surrogate Model Training
- Use a modified ResNet-18 to predict embeddings from images.
- Train using MSE loss to match target outputs.
The surrogate (stolen) model we used is a modified ResNet-18:

```python
base = models.resnet18(weights=None)
base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # remove initial large conv
base.maxpool = nn.Identity()  # remove unnecessary pooling
self.encoder = nn.Sequential(*list(base.children())[:-1])  # remove classifier
self.fc = nn.Linear(512, 1024)  # project to target dimension
```

### 5. Export & Submit
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
