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
import json
import gc
from dotenv import load_dotenv
from server_check import check_server

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Make predictions using a trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--image', type=str, help='Path to image file or image ID to predict')
    parser.add_argument('--images', type=str, help='Comma-separated list of image paths to predict in batch')
    parser.add_argument('--label', type=str, help='Label name (used if predicting by image ID)')
    parser.add_argument('--api-url', type=str, default=os.getenv("API_URL", "http://localhost:5000"),
                        help='API URL for the dataset server')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--batch', action='store_true', help='Enable batch prediction for all images of a label')
    parser.add_argument('--json-output', action='store_true', help='Output results as JSON')
    parser.add_argument('--quiet', action='store_true', help='Suppress debug output and only print JSON')
    return parser.parse_args()

# Create a function to print to stderr for debug messages when using JSON output
def debug_print(message):
    print(message, file=sys.stderr)

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
    
    # Get the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint with weights_only=True to avoid FutureWarning
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Create model with the same architecture
    model = create_model()
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to the device
    model = model.to(device)
    
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
        
        # Clear the image object to free memory
        image = None
        
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_multiple_images(model, image_paths, min_rating, max_rating, img_size, api_url=None):
    """Make predictions for multiple images at once - memory optimized version"""
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Process images in small batches to reduce memory usage
    batch_size = 4  # Reduced batch size
    
    for i in range(0, len(image_paths), batch_size):
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        batch = image_paths[i:i+batch_size]
        
        for image_path in batch:
            # Skip files that no longer exist (might have been deleted by Node.js)
            if not os.path.exists(image_path):
                results[image_path] = None
                continue
                
            try:
                # Load and prepare image
                image_tensor = prepare_image(image_path, img_size)
                
                if image_tensor is None:
                    results[image_path] = None
                    continue
                
                # Move to device and make prediction
                image_tensor = image_tensor.to(device)
                with torch.no_grad():
                    output = model(image_tensor)
                
                # Convert normalized output back to rating scale
                prediction = output.item() * (max_rating - min_rating) + min_rating
                results[image_path] = prediction
                
                # Clean up tensors to free memory
                del image_tensor, output
                
            except Exception as e:
                print(f"Error predicting image {image_path}: {e}")
                results[image_path] = None
                
        # Force Python garbage collection after each mini-batch
        gc.collect()
    
    return results

def main():
    args = parse_arguments()
    
    # Configure output based on JSON mode
    print_fn = debug_print if args.json_output and not args.quiet else print
    
    # Check if the server is running when we need it
    if args.batch or (args.image and not os.path.exists(args.image)) or \
       (args.images and any(not os.path.exists(img) for img in args.images.split(','))):
        if not check_server(args.api_url):
            print_fn(f"Failed to connect to server at {args.api_url}")
            return 1
    
    # Load the model
    try:
        model, min_rating, max_rating = load_model(args.model)
        print_fn(f"Model loaded successfully from {args.model}")
        print_fn(f"Rating range: {min_rating:.1f} to {max_rating:.1f}")
    except Exception as e:
        print_fn(f"Error loading model: {e}")
        return 1
    
    # Device is already set in load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_fn(f"Using device: {device}")
    
    # Multiple images prediction mode
    if args.images:
        image_paths = [path.strip() for path in args.images.split(',')]
        print_fn(f"Processing {len(image_paths)} images...")
        
        # Check for existing files
        image_paths = [path for path in image_paths if os.path.exists(path)]
        
        if not image_paths:
            print_fn("No valid image files to process")
            if args.json_output:
                print(json.dumps({"predictions": {}}))
            return 0
        
        predictions = predict_multiple_images(model, image_paths, min_rating, max_rating, args.img_size, args.api_url)
        
        # Output results as JSON or plain text
        if args.json_output:
            # Print ONLY the JSON to stdout for parsing
            json_output = {
                "predictions": {img_path: float(f"{rating:.2f}") if rating is not None else None 
                              for img_path, rating in predictions.items()}
            }
            print(json.dumps(json_output))
        else:
            for img_path, rating in predictions.items():
                if rating is not None:
                    print(f"Image: {img_path}, Rating: {rating:.2f}")
                else:
                    print(f"Image: {img_path}, Rating: Failed")
    
    # Single image prediction mode
    elif args.image:
        # Use the multiple image prediction function for consistency
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            if args.json_output:
                print(json.dumps({"error": "File not found"}))
            return 1
            
        predictions = predict_multiple_images(model, [args.image], min_rating, max_rating, args.img_size, args.api_url)
        prediction = predictions.get(args.image)
        
        if prediction is not None:
            if args.json_output:
                print(json.dumps({"prediction": float(f"{prediction:.2f}")}))
            else:
                # Make sure this format matches what we're looking for in predictImage.js
                print(f"Predicted rating: {prediction:.2f}")
                # Also print in a different format as backup 
                print(f"Rating={prediction:.2f}")
        else:
            if args.json_output:
                print(json.dumps({"error": "Failed to make prediction"}))
            else:
                print("Failed to make prediction")
            return 1
    else:
        print("Error: Either --image, --images, or --batch with --label must be specified")
        return 1
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
