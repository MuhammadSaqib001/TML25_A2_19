# ======================================
# stealing_query.py (store representations)
# ======================================
import os, io, sys, json, base64, pickle, requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from typing import Tuple
import time
import torchvision.transforms as T
import torchvision.transforms.functional as TF


token = "54614611"
chunk_size = 1000
max_queries = 100000

answer = {'seed': 18632002, 'port': '9054'}
# save the values
seed = int(answer['seed'])
port = str(answer['port'])

print(f"Using SEED={seed}, PORT={port}")

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class AugmentedData:
    pass

# Load dataset
data_simple = torch.load("ModelStealingPub.pt", weights_only=False)
data_aug = torch.load("ModelStealingPub_augmented.pt", weights_only=False)

imgs = data_simple.imgs + data_aug.imgs

mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])


# De-normalize to bring back to [0,1] range for conversion to PIL
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

indices = np.arange(len(imgs))
# Shuffle indices reproducibly
rng = np.random.default_rng(seed)
shuffled_indices = indices.copy()
rng.shuffle(shuffled_indices)

def query_api(img_list, port):
    url = f"http://34.122.51.94:{port}/query"
    image_data = []
    for img in img_list:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_data.append(b64)

    payload = json.dumps(image_data)
    resp = requests.get(url, files={"file": payload}, headers={"token": token})
    if resp.status_code == 200:
        return resp.json()["representations"]
    else:
        raise RuntimeError(f"API error: {resp.status_code}, {resp.text}")


all_reps, all_idx = [], []
# Process in shuffled order
for start in tqdm(range(0, min(len(imgs), max_queries), chunk_size)):
    batch_idx = shuffled_indices[start:start+chunk_size]
    batch_imgs = [imgs[i] for i in batch_idx]

    reps = query_api(batch_imgs, port)

    all_reps.extend(reps)
    all_idx.extend(batch_idx.tolist())

    # Save progress
    with open("embeddings.pickle", "wb") as f:
        pickle.dump({"indices": all_idx, "reps": all_reps}, f)

    # Wait 60 seconds before next API call
    print("Waiting for 60 seconds before the next call...")
    time.sleep(60)