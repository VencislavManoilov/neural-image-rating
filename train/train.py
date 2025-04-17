import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import io
import requests
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Parameters
API_URL = os.getenv("API_URL", "http://localhost:5000")  # Get from env or use default
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224

# Custom Dataset
class RatingDataset(Dataset):
    def __init__(self, df, api_url, transform=None):
        self.df = df
        self.api_url = api_url
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        
        # Fetch image from API
        img_url = f"{self.api_url}/get/{img_name}"
        response = requests.get(img_url)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch image {img_name}: {response.status_code}")
            
        # Convert response to image
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        rating = float(self.df.iloc[idx]['rating'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([rating], dtype=torch.float32)

# Load data from API
def fetch_labels():
    response = requests.get(f"{API_URL}/labels")
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch labels: {response.status_code}")
    
    data = response.json()
    return pd.DataFrame(data['labels'])

# Load labels data
df = fetch_labels()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataloaders
train_dataset = RatingDataset(train_df, API_URL, transform)
val_dataset = RatingDataset(val_df, API_URL, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model (ResNet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # Output: single float
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, ratings in train_loader:
            images, ratings = images.to(device), ratings.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")

# Eval
def evaluate():
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, ratings in val_loader:
            images = images.to(device)
            outputs = model(images).cpu().squeeze().numpy()
            preds.extend(outputs)
            truths.extend(ratings.squeeze().numpy())

    # Plot predictions vs ground truth
    plt.scatter(truths, preds, alpha=0.5)
    plt.plot([1,10], [1,10], '--', color='red')
    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Model Predictions")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
    evaluate()
