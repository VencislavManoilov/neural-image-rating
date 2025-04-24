import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
import os
import io
import requests
import numpy as np
import sys
from dotenv import load_dotenv
from server_check import check_server

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Make predictions using a trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--image', type=str, help='Path to image file or image ID to predict')
    parser.add_argument('--label', type=str, help='Label name (used if predicting by image ID)')
    parser.add_argument('--api-url', type=str, default=os.getenv("API_URL", "http://localhost:5000"),
                        help='API URL for the dataset server')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--batch', action='store_true', help='Enable batch prediction for all images of a label')
    return parser.parse_args()

def create_model():
    """Create the same model architecture as used in training"""
    model = models.efficientnet_b2(weights=None)
    
    # Replace classifier with the same architecture used in training
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
        nn.Sigmoid()
    )
    
    return model

def load_model(model_path):
    """Load a saved model from path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with the same architecture
    model = create_model()
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get the rating range
    min_rating = checkpoint.get('min_rating', 1.0)
    max_rating = checkpoint.get('max_rating', 10.0)
    
    return model, min_rating, max_rating

def prepare_image(image_path, img_size):
    """Load and prepare an image for prediction"""
    # Image preprocessing - same as validation transform in train.py
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # If image_path is a URL or API path, download it
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            # Otherwise load from local filesystem
            image = Image.open(image_path).convert('RGB')
            
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def fetch_image_from_api(image_id, api_url):
    """Fetch an image from the API by its ID"""
    try:
        response = requests.get(f"{api_url}/get/{image_id}")
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            print(f"Failed to fetch image {image_id}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching image {image_id}: {e}")
        return None

def fetch_label_images(label_name, api_url):
    """Fetch all images for a specific label"""
    try:
        response = requests.get(f"{api_url}/labels/{label_name}")
        if response.status_code != 200:
            print(f"Failed to fetch label data: Status {response.status_code}")
            return []
            
        data = response.json()
        return [item['image'] for item in data.get('labels', [])]
    except Exception as e:
        print(f"Error fetching label data: {e}")
        return []

def predict_single_image(model, image_path, min_rating, max_rating, img_size, api_url=None):
    """Make a prediction for a single image"""
    # Load image either from file or API
    if os.path.exists(image_path):
        image_tensor = prepare_image(image_path, img_size)
    elif api_url:
        # Try to fetch from API
        image = fetch_image_from_api(image_path, api_url)
        if image:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
        else:
            return None
    else:
        print(f"Image not found: {image_path}")
        return None
    
    if image_tensor is None:
        return None
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert normalized output back to rating scale
    prediction = output.item() * (max_rating - min_rating) + min_rating
    
    return prediction

def main():
    args = parse_arguments()
    
    # Check if the server is running when we need it
    if args.batch or not os.path.exists(args.image):
        if not check_server(args.api_url):
            print(f"Failed to connect to server at {args.api_url}")
            return 1
    
    # Load the model
    try:
        model, min_rating, max_rating = load_model(args.model)
        print(f"Model loaded successfully from {args.model}")
        print(f"Rating range: {min_rating:.1f} to {max_rating:.1f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Batch prediction mode
    if args.batch:
        if not args.label:
            print("Error: --label parameter is required for batch prediction")
            return 1
            
        print(f"Fetching images for label '{args.label}'...")
        image_ids = fetch_label_images(args.label, args.api_url)
        
        if not image_ids:
            print(f"No images found for label '{args.label}'")
            return 1
            
        print(f"Found {len(image_ids)} images. Starting batch prediction...")
        results = []
        
        for idx, image_id in enumerate(image_ids):
            prediction = predict_single_image(model, image_id, min_rating, max_rating, args.img_size, args.api_url)
            if prediction is not None:
                results.append((image_id, prediction))
                print(f"Image {idx+1}/{len(image_ids)}: {image_id} → Rating: {prediction:.2f}")
        
        # Print summary
        avg_rating = sum(r[1] for r in results) / len(results) if results else 0
        print(f"\nBatch prediction complete. Average rating: {avg_rating:.2f}")
        
    # Single image prediction mode
    elif args.image:
        prediction = predict_single_image(model, args.image, min_rating, max_rating, args.img_size, args.api_url)
        if prediction is not None:
            # Make sure this format matches what we're looking for in predictImage.js
            print(f"Predicted rating: {prediction:.2f}")
            # Also print in a different format as backup 
            print(f"Rating={prediction:.2f}")
        else:
            print("Failed to make prediction")
            return 1
    else:
        print("Error: Either --image or --batch with --label must be specified")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
