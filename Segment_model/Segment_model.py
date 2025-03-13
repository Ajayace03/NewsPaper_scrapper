import os
import yaml
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
import cv2
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a newspaper layout detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory with train/test/valid subfolders')
    parser.add_argument('--output_dir', type=str, default='newspaper_detection', help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--model_size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8 model size')
    return parser.parse_args()

def verify_dataset_structure(data_dir):
    """Verify the dataset structure and return counts of images in each split"""
    required_dirs = ['train', 'test', 'valid']
    
    # Check if all required directories exist
    for d in required_dirs:
        if not os.path.exists(os.path.join(data_dir, d)):
            raise ValueError(f"Required directory {d} not found in {data_dir}")
        if not os.path.exists(os.path.join(data_dir, d, 'images')):
            raise ValueError(f"Images directory not found in {data_dir}/{d}")
        if not os.path.exists(os.path.join(data_dir, d, 'labels')):
            raise ValueError(f"Labels directory not found in {data_dir}/{d}")
    
    train_count = len([f for f in os.listdir(os.path.join(data_dir, 'train', 'images')) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))])
    test_count = len([f for f in os.listdir(os.path.join(data_dir, 'test', 'images')) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))])
    valid_count = len([f for f in os.listdir(os.path.join(data_dir, 'valid', 'images')) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return train_count, test_count, valid_count

def get_class_names_from_labels(data_dir):
    """Extract class names and IDs from annotation files"""
    class_ids = set()
    
  
    label_dir = os.path.join(data_dir, 'train', 'labels')
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) >= 5:  
                        class_id = int(parts[0])
                        class_ids.add(class_id)
    
    newspaper_classes = {
        0: 'article',
        1: 'banner',
        2: 'headline',
        3: 'image',
        4: 'subheadline'
    }
    

    class_names = {}
    for class_id in sorted(class_ids):
        class_names[class_id] = newspaper_classes.get(class_id, f'class_{class_id}')
        
    return class_names

def create_dataset_yaml(data_dir, output_dir, class_names):
    """Create a YAML configuration file for the dataset"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    yaml_content = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'valid/images',  
        'test': 'test/images',
        'names': class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    return yaml_path

def train_model(yaml_path, output_dir, model_size='m', epochs=100, batch_size=16, img_size=640):
    """Train the YOLOv8 model"""

    model = YOLO(f'yolov8{model_size}.pt')
    

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=output_dir,
        name='train',
        device=0 if torch.cuda.is_available() else 'cpu',
        plots=True
    )
    
    return model, results

def validate_model(model, data_yaml, output_dir):
    """Validate the trained model"""
    val_results = model.val(data=data_yaml, project=output_dir, name='validation')
    return val_results

def test_model(model, test_dir, output_dir):
    """Test the model on the test set and visualize results"""
    test_img_dir = os.path.join(test_dir, 'images')
    results_dir = os.path.join(output_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    
    print(f"Processing test images from {test_img_dir}...")
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(test_images, desc="Testing images"):
        img_path = os.path.join(test_img_dir, img_file)
        
        # Perform prediction
        results = model.predict(img_path, save=True, project=results_dir, name='')
        
        # Save visualization
        result_img = results[0].plot()
        cv2.imwrite(os.path.join(results_dir, f"viz_{img_file}"), result_img)
    
    print(f"Test results saved to {results_dir}")

def export_model(model, output_dir):
    """Export the model to ONNX format for deployment"""
    print("Exporting model to ONNX format...")
    model.export(format='onnx', project=output_dir, name='export')
    
    print("Exporting model to TorchScript format...")
    model.export(format='torchscript', project=output_dir, name='export')
    
    print(f"Exported models saved to {os.path.join(output_dir, 'export')}")

def main():
    args = parse_args()
    
    print(f"Setting up newspaper layout detection pipeline...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify dataset structure
    print("Verifying dataset structure...")
    train_count, test_count, valid_count = verify_dataset_structure(args.data_dir)
    print(f"Dataset verified: {train_count} train, {valid_count} validation, {test_count} test images")
    
    # Get class names
    class_names = get_class_names_from_labels(args.data_dir)
    print(f"Detected classes: {class_names}")
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.data_dir, args.output_dir, class_names)
    print(f"Created dataset configuration at {yaml_path}")
    
    # Train the model
    print(f"Starting model training with YOLOv8{args.model_size}...")
    model, train_results = train_model(
        yaml_path,
        args.output_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Validate the model
    print("Validating model...")
    val_results = validate_model(model, yaml_path, args.output_dir)
    
    # Test the model
    print("Testing model and generating visualizations...")
    test_model(model, os.path.join(args.data_dir, 'test'), args.output_dir)
    
    # Export the model
    print("Exporting model for deployment...")
    export_model(model, args.output_dir)
    
    print(f"Training complete! Results saved to {args.output_dir}")
    print(f"Model exports saved to {os.path.join(args.output_dir, 'export')}")
    print(f"Run inference with: yolo predict model={os.path.join(args.output_dir, 'export', 'model.onnx')} source=/path/to/image.jpg")

if __name__ == "__main__":
    main()