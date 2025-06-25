import pickle, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import onnxruntime as ort
from typing import Tuple
import numpy as np
import torchvision.models as models

BATCH_SIZE = 512 
EPOCHS = 40


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class StealDataset(Dataset):
    def __init__(self, indices, images, reps, transform=None):
        self.indices = indices
        self.images = images
        for i, img in enumerate(self.images):
            img = np.array(img)
            # If grayscale: shape will be (H, W)
            if len(img.shape) == 2:
                # Convert grayscale to RGB by stacking
                img = np.stack([img] * 3, axis=-1)
            self.images[i] = Image.fromarray(img)
        self.reps = reps
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img = self.images[self.indices[i]]  # PIL image
        if self.transform:
            img = self.transform(img)
        return img, self.reps[i]


class StolenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # remove initial large conv
        base.maxpool = nn.Identity()  # remove unnecessary pooling
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # remove classifier
        self.fc = nn.Linear(512, 1024)  # project to target dimension

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Training script starts here 
with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)
indices = data["indices"]
reps = torch.tensor(data["reps"], dtype=torch.float32)

data_simple = torch.load("ModelStealingPub.pt", weights_only=False)
data_aug = torch.load("ModelStealingPub_augmented.pt", weights_only=False)

imgs = data_simple.imgs + data_aug.imgs

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.2980, 0.2962, 0.2987], [0.2886, 0.2875, 0.2889])
])

dataset = StealDataset(indices, imgs, reps, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = StolenEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x,y in dataloader:
        x,y = x.to(model.encoder[0].weight.device), y.to(model.encoder[0].weight.device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}, Loss: {total/len(dataset):.4f}")

torch.onnx.export(model, torch.randn(1,3,32,32).to(model.encoder[0].weight.device), "stolen_encoder.onnx", input_names=["x"], output_names=["z"], opset_version=12)
sess = ort.InferenceSession("stolen_encoder.onnx")
print("ONNX check output shape:", sess.run(None,{"x":np.random.randn(1,3,32,32).astype(np.float32)})[0].shape)
