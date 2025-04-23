import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd # type: ignore
import os
import io
import requests
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split # type: ignore
import matplotlib.pyplot as plt
import time
from dotenv import load_dotenv # type: ignore
from server_check import check_server

# Load environment variables from .env file
load_dotenv()

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train model on labeled images')
    parser.add_argument('--label', type=str, help='Label name to train on')
    parser.add_argument('--api-url', type=str, default=os.getenv("API_URL", "http://localhost:5000"),
                        help='API URL for the dataset server')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for training')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='./models', help='Directory to save model')
    return parser.parse_args()

# Key parameters - will be overridden by command line args
API_URL = os.getenv("API_URL", "http://localhost:5000")
BATCH_SIZE = 16
EPOCHS = 20        
LR = 3e-4           
IMG_SIZE = 224      
NUM_WORKERS = 2   
MODEL_TYPE = "regression"
LABEL_NAME = None
OUTPUT_DIR = "./models"

# Move all global variables and initialization into a function
def init_training(args):
    global API_URL, BATCH_SIZE, EPOCHS, LR, IMG_SIZE, NUM_WORKERS, LABEL_NAME, OUTPUT_DIR
    
    # Override globals with command line arguments
    API_URL = args.api_url
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    IMG_SIZE = args.img_size
    NUM_WORKERS = args.workers
    LABEL_NAME = args.label
    OUTPUT_DIR = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parameters dictionary for convenience
    config = {
        'api_url': API_URL,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'img_size': IMG_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': True,
        'label_name': LABEL_NAME,
        'output_dir': OUTPUT_DIR
    }
    return config

# Simplified RatingDataset class
class RatingDataset(Dataset):
    _image_cache = {}  # Shared cache for all instances
    
    def __init__(self, df, api_url, transform=None, cache_dir="./image_cache"):
        self.df = df
        self.api_url = api_url
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Create cache directory if needed
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        rating = float(self.df.iloc[idx]['rating'])  # Already normalized in fetch_labels
        
        # Try to get image from cache or load it
        try:
            # First check memory cache
            if img_name in self._image_cache:
                image = self._image_cache[img_name]
            else:
                # Then check disk cache
                cache_path = os.path.join(self.cache_dir, img_name)
                if os.path.exists(cache_path):
                    image = Image.open(cache_path).convert('RGB')
                    self._image_cache[img_name] = image
                else:
                    # Fetch from API as last resort
                    response = requests.get(f"{self.api_url}/get/{img_name}")
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content)).convert('RGB')
                        self._image_cache[img_name] = image
                        # Save to disk cache
                        image.save(cache_path)
                    else:
                        # Return gray placeholder if image can't be loaded
                        image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor([rating], dtype=torch.float32)
            
        except Exception as e:
            # Return placeholder on error with minimal logging
            if os.environ.get('DEBUG'):
                print(f"Error loading image {img_name}: {e}")
                
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([rating], dtype=torch.float32)

# Load data from API for a specific label
def fetch_labels():
    print(f"Fetching data for label '{LABEL_NAME}' from API...")
    if LABEL_NAME:
        response = requests.get(f"{API_URL}/labels/{LABEL_NAME}")
    else:
        response = requests.get(f"{API_URL}/labels")
        
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch labels: {response.status_code}")
    
    data = response.json()
    df = pd.DataFrame(data['labels'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Get rating range
    min_rating = float(df['rating'].min())
    max_rating = float(df['rating'].max())
    print(f"Rating range: {min_rating:.1f} to {max_rating:.1f}")
    
    # Normalize ratings to [0, 1] range
    df['original_rating'] = df['rating'].copy()
    df['rating'] = (df['rating'] - min_rating) / (max_rating - min_rating)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Dataset: {len(df)} total, {len(train_df)} training, {len(val_df)} validation")
    
    return train_df, val_df, min_rating, max_rating

# Data transforms
def get_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 64, img_size + 64)),  # Larger resize for more variation
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),  # More rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Increased color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Add affine transforms
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Add perspective changes
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Add random erasing for robustness
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Add worker initialization function
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create classification model
def create_classification_model(num_classes=10):
    print("Creating classification model...")
    # Use a better model for more accurate classifications
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Replace final fully connected layer with a more powerful classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        
        nn.Linear(256, num_classes)  # No activation - will use CrossEntropyLoss
    )
    
    print(f"Model created with EfficientNet backbone for {num_classes}-class classification")
    return model

# Create a better regression model specifically for image rating
def create_regression_model():
    print("Creating specialized image rating regression model...")
    
    # Use a more powerful backbone for better feature extraction
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    
    # Replace final fully connected layer with a more nuanced regression head
    # that can better distinguish between similar ratings
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.4),
        
        nn.Linear(512, 256), 
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.4),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.3),
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),
        
        # Final layer with sigmoid activation
        nn.Linear(64, 1),
        nn.Sigmoid()  # Output in [0,1] range
    )
    
    print("Enhanced regression model created with deeper architecture")
    return model

# Fixed training function with minimal logging, proper epoch timing, and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, min_rating, max_rating, scheduler=None, epochs=20):
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early stopping parameters
    patience = 10  # Number of epochs to wait after last improvement
    early_stop_counter = 0
    
    # Enable mixed precision if available
    scaler = torch.cuda.amp.GradScaler() if hasattr(torch.cuda, 'amp') else None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for images, ratings in train_loader:
            images = images.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, ratings)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            batch_count += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()  # Ensure model is in evaluation mode
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for images, ratings in val_loader:
                images = images.to(device, non_blocking=True)
                ratings = ratings.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, ratings)
                
                val_loss += loss.item()
                batch_count += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / batch_count if batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Update learning rate if using ReduceLROnPlateau scheduler
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print status with minimal logging (only once per epoch)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'min_rating': min_rating,
                'max_rating': max_rating,
                'best_val_loss': best_val_loss,
            }, "best_model.pth")
            print(f" New best model saved (val loss: {avg_val_loss:.4f})")
            # Reset early stopping counter
            early_stop_counter = 0
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs (best val loss: {best_val_loss:.4f})")
        
        # Check if we should stop early
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'min_rating': min_rating,
        'max_rating': max_rating,
        'final_val_loss': avg_val_loss,
    }, "model_final.pth")
    
    print("\nTraining completed!")
    return train_losses, val_losses

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, ratings in val_loader:
            images, ratings = images.to(device, non_blocking=True), ratings.to(device, non_blocking=True)
            
            # Simpler version without device_type parameter
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, ratings)
                
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

# Clean evaluation function that properly handles plots
def evaluate_model(model, val_loader, min_rating, max_rating):
    print("\nEvaluating model performance...")
    model.eval()
    preds = []
    truths = []
    
    with torch.no_grad():
        for images, ratings in val_loader:
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            
            # Denormalize predictions and ground truth
            outputs_denorm = outputs.cpu().numpy() * (max_rating - min_rating) + min_rating
            ratings_denorm = ratings.numpy() * (max_rating - min_rating) + min_rating
            
            # Collect results
            preds.extend(outputs_denorm.flatten())
            truths.extend(ratings_denorm.flatten())
    
    # Convert to numpy arrays
    preds = np.array(preds)
    truths = np.array(truths)
    
    # Calculate metrics
    mse = np.mean((preds - truths) ** 2)
    mae = np.mean(np.abs(preds - truths))
    
    print(f"Model Performance - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Create plot with better appearance
    # plt.figure(figsize=(8, 8))
    plt.scatter(truths, preds, alpha=0.5, s=30, c='royalblue', edgecolors='navy')
    
    # Add perfect prediction line
    min_val = min(min(truths), min(preds))
    max_val = max(max(truths), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
    
    # Adjust plot appearance
    plt.xlabel("True Rating", fontsize=12)
    plt.ylabel("Predicted Rating", fontsize=12)
    plt.title(f"Model Predictions (MSE: {mse:.4f}, MAE: {mae:.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure with high quality
    # plt.savefig('predictions.png', dpi=300)
    plt.show()
    
    return mse, mae

# Evaluation function for classification model
def evaluate_classification_model(model, val_loader, min_rating, max_rating):
    print("\nEvaluating classification model performance...")
    model.eval()
    all_preds = []
    all_truth = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Get predicted class scores
            outputs = model(images)
            
            # Get predicted class indices
            _, predicted_classes = torch.max(outputs.data, 1)
            
            # Track accuracy
            total += labels.size(0)
            correct += (predicted_classes == labels).sum().item()
            
            # Store predictions and ground truth for detailed evaluation
            # Convert class indices back to ratings (add 1 since classes are 0-indexed)
            pred_ratings = (predicted_classes.cpu().numpy() + 1).astype(float)
            true_ratings = (labels.cpu().numpy() + 1).astype(float)
            
            all_preds.extend(pred_ratings)
            all_truth.extend(true_ratings)
    
    # Calculate metrics
    accuracy = 100 * correct / total
    all_preds = np.array(all_preds)
    all_truth = np.array(all_truth)
    mse = np.mean((all_preds - all_truth) ** 2)
    mae = np.mean(np.abs(all_preds - all_truth))
    
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Create plot to visualize predictions
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with jitter for better visibility
    jitter = np.random.normal(0, 0.1, size=len(all_preds))
    plt.scatter(all_truth + jitter, all_preds, alpha=0.5, s=50, c='royalblue', edgecolors='navy')
    
    # Add perfect prediction line
    plt.plot([1, 10], [1, 10], '--', color='red', linewidth=2)
    
    # Set axis limits and labels
    plt.xlim(0.5, 10.5)
    plt.ylim(0.5, 10.5)
    plt.xticks(range(1, 11))
    plt.yticks(range(1, 11))
    plt.xlabel("True Rating", fontsize=12)
    plt.ylabel("Predicted Rating", fontsize=12)
    plt.title(f"Model Predictions (Accuracy: {accuracy:.1f}%, MSE: {mse:.2f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show figure
    # plt.savefig('predictions.png', dpi=300)
    plt.show()
    
    return accuracy, mse, mae

# Main function to orchestrate the entire process
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize training parameters
    config = init_training(args)
    
    # Verify server is running
    if not check_server(API_URL):
        print(f"Failed to connect to server at {API_URL}")
        return
    
    # Set up device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model output directory with label name
    model_dir = os.path.join(OUTPUT_DIR, LABEL_NAME) if LABEL_NAME else OUTPUT_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and prepare data
    train_df, val_df, min_rating, max_rating = fetch_labels()
    
    # Get data transforms
    train_transform, val_transform = get_transforms(IMG_SIZE)
    
    # Create datasets - always use regression approach
    cache_dir = "./image_cache"
    train_dataset = RatingDataset(train_df, API_URL, train_transform, cache_dir=cache_dir)
    val_dataset = RatingDataset(val_df, API_URL, val_transform, cache_dir=cache_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create regression model
    model = create_regression_model().to(device)
    
    # Use combination of MSE and Huber loss for better performance
    criterion = nn.SmoothL1Loss()
    
    # Use a more balanced optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler - add verbose parameter if desired
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, min_rating, max_rating, scheduler, 
        epochs=EPOCHS
    )
    
    # Save model files to the label-specific directory
    model_path = os.path.join(model_dir, "model_final.pth")
    best_model_path = os.path.join(model_dir, "best_model.pth")
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'min_rating': min_rating,
        'max_rating': max_rating,
        'label': LABEL_NAME,
    }, model_path)
    
    # Copy the best model if it exists in the current directory
    if os.path.exists("best_model.pth"):
        import shutil
        shutil.copy("best_model.pth", best_model_path)
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {LABEL_NAME}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, 'learning_curve.png'))
    plt.close()
    
    # Evaluate final model
    mse, mae = evaluate_model(model, val_loader, min_rating, max_rating)
    
    # Save evaluation metrics
    with open(os.path.join(model_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Label: {LABEL_NAME}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"Min Rating: {min_rating}\n")
        f.write(f"Max Rating: {max_rating}\n")
    
    print(f"Training and evaluation completed successfully for label {LABEL_NAME}!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Set global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run main process
    main()
