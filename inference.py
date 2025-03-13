import os
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from transformers import pipeline
from scipy.signal import find_peaks
from ocr_cleaning import process_newspaper_with_llm

# Load environment variables from .env file for API keys
load_dotenv()

# Initialize EasyOCR reader globally
reader = None

def initialize_ocr(languages=['en']):
    """Initialize the EasyOCR reader with specified languages"""
    global reader
    if reader is None:
        try:
            reader = easyocr.Reader(
                languages,
                gpu=torch.cuda.is_available(),  # Use GPU if available
                model_storage_directory=os.path.join(os.path.expanduser("~"), '.EasyOCR')
            )
            print(f"Initialized EasyOCR with languages: {languages}")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            return None
    return reader

def parse_args():
    parser = argparse.ArgumentParser(description='Extract newspaper content hierarchically, process with LLM, and output JSON')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLO model weights')
    parser.add_argument('--image', type=str, help='Path to a single image for inference')
    parser.add_argument('--input_dir', type=str, help='Path to directory with images')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for JSON results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--languages', type=str, default='en', help='Languages for OCR, comma-separated (e.g., en,fr)')
    parser.add_argument('--save_visualization', action='store_true', help='Save visualization of detected boxes')
    parser.add_argument('--auto_columns', action='store_true', help='Automatically detect columns')
    parser.add_argument('--num_columns', type=int, default=None, help='Number of columns to extract (overrides auto detection)')
    return parser.parse_args()

def load_model(model_path):
    """Load the trained YOLO model"""
    model = YOLO(model_path)
    return model

def extract_text_from_region(img, box):
    """Extract text from image region using EasyOCR"""
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Ensure coordinates are within image boundaries
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Skip if region is too small
    if x2 <= x1 or y2 <= y1 or (x2-x1) < 5 or (y2-y1) < 5:
        return ""
    
    try:
        # Crop the region from the image
        region = img[y1:y2, x1:x2]
        
        # Check if region is valid
        if region.size == 0:
            return ""
        
        # Ensure the OCR reader is initialized
        global reader
        if reader is None:
            reader = initialize_ocr()
            if reader is None:
                return ""
        
        # Perform OCR on the region
        results = reader.readtext(region)
        
        # Extract and concatenate the recognized text
        extracted_text = " ".join([result[1] for result in results])
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def extract_text_by_columns(img, num_columns=None, overlap_threshold=0.3):
    """
    Extract text from image in a column-based manner
    
    Args:
        img: Input image
        num_columns: Number of columns to detect (if None, auto-detect)
        overlap_threshold: Threshold for determining column overlap
        
    Returns:
        List of dictionaries, each containing column text and position
    """
    h, w = img.shape[:2]
    
    # Ensure the OCR reader is initialized
    global reader
    if reader is None:
        reader = initialize_ocr()
        if reader is None:
            return []
    
    # If num_columns not specified, auto-detect
    if num_columns is None:
        num_columns, boundaries = analyze_column_layout(img)
    else:
        # Create evenly spaced column boundaries
        boundaries = [int(i * w / num_columns) for i in range(num_columns + 1)]
    
    # Step 1: Get all text boxes from the image
    results = reader.readtext(img)
    
    if not results:
        return []
    
    # Step 2: Group text boxes by column
    columns = []
    for i in range(num_columns):
        left_bound = boundaries[i]
        right_bound = boundaries[i + 1]
        
        # Allow for some overlap between columns
        adjusted_left = max(0, left_bound - (boundaries[i+1] - boundaries[i]) * overlap_threshold)
        adjusted_right = min(w, right_bound + (boundaries[i+1] - boundaries[i]) * overlap_threshold)
        
        # Filter text boxes that belong to this column
        column_boxes = []
        for box, text, conf in results:
            # Calculate box center x
            box_center_x = sum(p[0] for p in box) / len(box)
            
            # Check if the center of the box is within column bounds
            if adjusted_left <= box_center_x <= adjusted_right:
                column_boxes.append((box, text, conf))
        
        # Sort boxes by vertical position (top to bottom)
        column_boxes.sort(key=lambda item: min(p[1] for p in item[0]))
        
        # Extract text from each box and join
        column_text = [text for _, text, _ in column_boxes]
        
        columns.append({
            'column_index': i,
            'bounds': (left_bound, right_bound),
            'text': ' '.join(column_text),
            'boxes': [box for box, _, _ in column_boxes],
            'raw_results': column_boxes
        })
    
    return columns

def analyze_column_layout(img):
    """
    Analyze image to determine the likely column layout
    
    Args:
        img: Input image
        
    Returns:
        Estimated number of columns and column boundaries
    """
    h, w = img.shape[:2]
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Sum edges vertically to find potential column divisions
    vertical_projection = np.sum(edges, axis=0)
    
    # Smooth the projection
    kernel_size = max(w//30, 5)  # Adjust kernel size based on image width
    smoothed = np.convolve(vertical_projection, np.ones(kernel_size)/float(kernel_size), mode='same')
    
    # Find valleys (potential column boundaries)
    valleys, _ = find_peaks(-smoothed, distance=w//6)
    
    # If no clear valleys are found, analyze text distribution
    if len(valleys) < 1:
        # Run OCR to get text boxes
        global reader
        if reader is None:
            reader = initialize_ocr()
            
        if reader:
            results = reader.readtext(img)
            if results:
                # Extract center x-coordinates of text boxes
                x_centers = [sum(p[0] for p in box)/len(box) for box, _, _ in results]
                
                # Create histogram of x-centers
                hist, bin_edges = np.histogram(x_centers, bins=min(20, len(x_centers)//2 + 1))
                
                # Find peaks in histogram to identify column centers
                peaks, _ = find_peaks(hist, height=max(hist)/5, distance=w/10)
                
                if len(peaks) >= 1:
                    # Calculate column boundaries from peaks
                    peak_x = [bin_edges[p] + (bin_edges[p+1] - bin_edges[p])/2 for p in peaks]
                    boundaries = [0] + [int((peak_x[i] + peak_x[i+1])/2) for i in range(len(peak_x)-1)] + [w]
                    return len(boundaries) - 1, boundaries
    
    # Add left and right boundaries
    boundaries = [0] + list(valleys) + [w]
    boundaries.sort()
    
    # Determine number of columns
    if len(boundaries) <= 2:
        num_columns = 1
    else:
        num_columns = len(boundaries) - 1
    
    return num_columns, boundaries

def process_article_in_columns(img, headline_box, num_columns=None):
    """
    Process an article by extracting text in columns below the headline
    
    Args:
        img: Input image
        headline_box: Bounding box of the headline [x1, y1, x2, y2]
        num_columns: Number of columns (if None, auto-detect)
        
    Returns:
        Structured article content with columns
    """
    x1, y1, x2, y2 = [int(coord) for coord in headline_box]
    
    # Define region below the headline
    h, w = img.shape[:2]
    content_y1 = int(y2)
    content_y2 = int(min(h, h - 1))  # Look until bottom of image
    
    # Crop the image to focus on content below the headline
    content_region = img[content_y1:content_y2, 0:w]
    
    # Extract text in columns
    columns = extract_text_by_columns(content_region, num_columns)
    
    # Clean and enhance text for each column
    for i, column in enumerate(columns):
        if column['text']:
            # Clean the text
            column['text'] = clean_ocr_text(column['text'])
            # Enhance with LLM
            column['text'] = enhance_text_with_llm(column['text'])
    
    return columns

# Initialize the text correction pipeline globally to avoid reloading it multiple times
text_corrector = None

def initialize_text_corrector():
    """Initialize the text correction pipeline with a Hugging Face model"""
    global text_corrector
    if text_corrector is None:
        try:
            # Use T5 model for text2text generation - good for correction/summarization
            text_corrector = pipeline(
                "text2text-generation",
                model="facebook/bart-large-cnn",  # Small model for faster processing
                device=-1 if not torch.cuda.is_available() else 0  # Use GPU if available
            )
            print("Initialized Hugging Face text correction model")
        except Exception as e:
            print(f"Error initializing text correction model: {e}")
            # Fallback to a simpler model if the first one fails
            try:
                text_corrector = pipeline(
                    "text2text-generation",
                    model="t5-small",
                    device=-1 if not torch.cuda.is_available() else 0
                )
                print("Initialized fallback text correction model")
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
                return None
    return text_corrector

def enhance_text_with_llm(text):
    """Use Hugging Face models to enhance and clean up extracted OCR text"""
    if not text or not text.strip():
        return ""
    
    try:
        # Ensure the text corrector is initialized
        corrector = initialize_text_corrector()
        if corrector is None:
            return clean_ocr_text(text)  # Fallback to rule-based cleaning
        
        # Prepare prompt
        prompt = f"Fix OCR errors in this text: {text}"
        
        # Run inference with the model
        result = corrector(
            prompt,
            max_length=len(text.split()) + 20,  # Allow for some expansion
            min_length=max(5, len(text.split()) - 10),  # Allow for some compression
            do_sample=False,  # Deterministic output
            num_return_sequences=1
        )
        
        # Extract the generated text
        enhanced_text = result[0]['generated_text'].strip()
        
        # If the model returns something empty or much shorter, use the original
        if not enhanced_text or len(enhanced_text) < len(text) / 2:
            return clean_ocr_text(text)
        
        return enhanced_text
    except Exception as e:
        print(f"Error using LLM for text enhancement: {e}")
        return clean_ocr_text(text)  # Fallback to rule-based cleaning

def clean_ocr_text(text):
    """Clean OCR text using simple rule-based approaches"""
    if not text or not text.strip():
        return ""
    
    # Remove common OCR errors
    text = re.sub(r'[|]', 'I', text)  # Replace pipe with I
    text = re.sub(r'[\[\]]', '', text)  # Remove brackets
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'([.,!?:;])\s*', r'\1 ', text)  # Add space after punctuation
    
    # Fix common OCR mistakes
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
    
    # Join hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
    
    return text.strip()

def is_close_to(box1, box2, vertical_threshold=1.5, horizontal_threshold=1.0):
    """Check if box1 is close to box2 using improved spatial relationship detection"""
    x1_1, y1_1, x2_1, y2_1 = [int(coord) for coord in box1]
    x1_2, y1_2, x2_2, y2_2 = [int(coord) for coord in box2]

    # Calculate centers
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2
    
    # Calculate box dimensions
    width1 = x2_1 - x1_1
    height1 = y2_1 - y1_1
    width2 = x2_2 - x1_2
    height2 = y2_2 - y1_2
    
    # Check if box1 is below box2 (for text blocks below headlines)
    if y1_1 >= y2_2:  # box1 starts after box2 ends (vertically)
        vertical_distance = y1_1 - y2_2
        max_allowed_v_distance = vertical_threshold * max(height1, height2)
        
        # Check horizontal overlap or proximity
        horizontal_overlap = (x1_1 <= x2_2 and x2_1 >= x1_2)
        if horizontal_overlap and vertical_distance <= max_allowed_v_distance:
            return True
    
    # Check if box1 is to the right of box2 (for multicolumn layout)
    if x1_1 >= x2_2:  # box1 starts after box2 ends (horizontally)
        horizontal_distance = x1_1 - x2_2
        max_allowed_h_distance = horizontal_threshold * max(width1, width2)
        
        # Check vertical alignment
        vertical_overlap = (y1_1 <= y2_2 and y2_1 >= y1_2)
        if vertical_overlap and horizontal_distance <= max_allowed_h_distance:
            return True
    
    return False

def visualize_detections(img, detections, output_path):
    """Save a visualization of the detected boxes"""
    # Create a copy of the image to draw on
    vis_img = img.copy()
    
    # Convert to BGR for OpenCV
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Define colors for different classes
    colors = {
        'headline': (0, 0, 255),       # Red
        'subheadline': (0, 255, 0),    # Green
        'text_block': (255, 0, 0),     # Blue
        'text': (255, 0, 0)            # Blue (same as text_block)
    }
    
    # Draw boxes and labels
    for detection in detections:
        class_name = detection['class'].lower()
        box = detection['box']
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get color for class (default to yellow if class not found)
        color = colors.get(class_name, (0, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{class_name}: {detection['confidence']:.2f}"
        cv2.putText(
            vis_img, label, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    # Save the visualization
    cv2.imwrite(output_path, vis_img)
    print(f"Saved visualization to {output_path}")


def generate_fallback_content(img, headline_box, num_columns=None):
    """Generate content for an article by looking below the headline using column-based approach"""
    x1, y1, x2, y2 = [int(coord) for coord in headline_box]
    
    # Define a region below the headline
    h, w = img.shape[:2]
    content_y1 = int(y2)
    content_y2 = int(min(h, y2 + 500))  # Look 500px below headline or to image boundary
    
    # Extract content using column-based approach
    content_region = img[content_y1:content_y2, 0:w]
    columns = extract_text_by_columns(content_region, num_columns)
    
    # Process each column
    for i, column in enumerate(columns):
        if column['text']:
            column['text'] = clean_ocr_text(column['text'])
            column['text'] = enhance_text_with_llm(column['text'])
    
    # Return both structured columns and combined text
    return {
        'columns': columns,
        'combined_text': " ".join([col['text'] for col in columns if col['text'].strip()])
    }
def visualize_columns(img, columns, output_path):
    """Visualize detected columns in the image"""
    vis_img = img.copy()
    
    # Convert to BGR for OpenCV
    if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    h, w = vis_img.shape[:2]
    
    # Draw column boundaries
    for i, column in enumerate(columns):
        left, right = column['bounds']
        
        # Convert to integer coordinates
        left, right = int(left), int(right)
        
        # Draw vertical lines for column boundaries
        cv2.line(vis_img, (left, 0), (left, h), (0, 255, 0), 2)
        cv2.line(vis_img, (right, 0), (right, h), (0, 255, 0), 2)
        
        # Label the column
        cv2.putText(
            vis_img, f"Column {i+1}", (left + 10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        
        # Draw boxes for text in this column
        for box in column.get('boxes', []):
            # Convert polygon points to rectangle
            points = np.array(box, dtype=np.int32)
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    
    # Save the visualization
    cv2.imwrite(output_path, vis_img)
    print(f"Saved column visualization to {output_path}")

def process_single_image(model, image_path, conf_threshold=0.05, ocr_languages=['en'], save_visualization=False, num_columns=None):
    """Process a single image and extract hierarchical article structure as JSON with column support"""
    # Initialize EasyOCR with specified languages
    initialize_ocr(ocr_languages)
    
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Run inference
    try:
        results = model(image_path, conf=conf_threshold)
        result = results[0]
    except Exception as e:
        print(f"Error running model inference: {e}")
        return None
    
    # Get the original image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image: {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # If num_columns not specified, auto-detect from the whole image
    if num_columns is None:
        detected_columns, _ = analyze_column_layout(img)
        print(f"Auto-detected {detected_columns} columns in the image")
        num_columns = detected_columns
    
    # Dictionary to store class names
    class_names = result.names
    
    # Lists to store detected objects
    headlines = []
    subheadlines = []
    text_blocks = []
    all_detections = []
    
    # Process detections and categorize them
    for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
        box_coords = box.tolist()
        class_id = int(cls)
        class_name = class_names[class_id]
        confidence = float(conf)
        
        # Extract text from the region
        text = extract_text_from_region(img, box_coords)
        
        # Clean the OCR text
        text = clean_ocr_text(text)
        
        # Store detection with text
        detection = {
            'id': i,
            'box': box_coords,
            'class': class_name,
            'confidence': confidence,
            'text': text
        }
        
        # Add to all detections list
        all_detections.append(detection)
        
        # Categorize by class
        if class_name.lower() == 'headline':
            headlines.append(detection)
        elif class_name.lower() == 'subheadline':
            subheadlines.append(detection)
        elif class_name.lower() in ['text_block', 'text']:
            text_blocks.append(detection)
    
    # Print detection stats for debugging
    print(f"Total detections: {len(all_detections)}")
    print(f"Headlines detected: {len(headlines)}")
    print(f"Subheadlines detected: {len(subheadlines)}")
    print(f"Text blocks detected: {len(text_blocks)}")
    
    # Save visualization if requested
    if save_visualization:
        output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else '.'
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        vis_output_path = os.path.join(output_dir, f"{base_filename}_visualization.jpg")
        visualize_detections(img, all_detections, vis_output_path)
    
    # Initialize the LLM text corrector
    initialize_text_corrector()
    
    # Sort elements by position (top to bottom, left to right)
    headlines.sort(key=lambda x: (x['box'][1], x['box'][0]))
    subheadlines.sort(key=lambda x: (x['box'][1], x['box'][0]))
    text_blocks.sort(key=lambda x: (x['box'][1], x['box'][0]))
    
    # Structure the text into articles
    articles = []
    
    # For each headline, find associated subheadlines and process content in columns
    for idx, headline in enumerate(headlines):
        article = {
            'article_number': idx + 1,
            'headline': headline['text'],
            'subheadlines': [],
            'columns': [],
            'content': "",
            'position': headline['box']
        }
        
        headline_box = headline['box']
        
        # Find subheadlines associated with this headline
        associated_subheadlines = []
        for subheadline in subheadlines:
            if is_close_to(subheadline['box'], headline_box):
                associated_subheadlines.append(subheadline['text'])
                subheadline['assigned'] = True
        
        article['subheadlines'] = associated_subheadlines
        
        # Process article content using column-based extraction
        columns = process_article_in_columns(img, headline_box, num_columns)
        
        # Store column information
        article['columns'] = []
        for i, column in enumerate(columns):
            article['columns'].append({
                'column_number': i + 1,
                'text': column['text']
            })
        
        # For backwards compatibility, also include all text as a single content field
        article['content'] = " ".join([col['text'] for col in article['columns'] if col['text'].strip()])
        
        # If no content was found via columns, try the fallback method
        if not article['content']:
            fallback_result = generate_fallback_content(img, headline_box, num_columns)
            
            # Update columns and content
            if fallback_result['columns']:
                article['columns'] = []
                for i, column in enumerate(fallback_result['columns']):
                    article['columns'].append({
                        'column_number': i + 1,
                        'text': column['text']
                    })
            
            article['content'] = fallback_result['combined_text']
        
        articles.append(article)
        
        # Save column visualization if requested
        if save_visualization and columns:
            output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else '.'
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            col_vis_path = os.path.join(output_dir, f"{base_filename}_article{idx+1}_columns.jpg")
            
            # Create a cropped image for this article
            x1, y1, x2, y2 = [int(coord) for coord in headline_box]
            article_img = img[y1:, 0:img.shape[1]]  # From headline to bottom
            
            # Fix: Pass the article_img as the first parameter
            visualize_columns(article_img, columns, col_vis_path)
    
    # Handle orphan text blocks (not associated with any headline)
    orphan_text_blocks = [tb for tb in text_blocks if not tb.get('assigned', False)]
    if orphan_text_blocks:
        # Extract the text from orphan blocks using column-based approach
        orphan_img = img.copy()
        
        # Find the bounding box containing all orphan blocks
        x_min = min(tb['box'][0] for tb in orphan_text_blocks)
        y_min = min(tb['box'][1] for tb in orphan_text_blocks)
        x_max = max(tb['box'][2] for tb in orphan_text_blocks)
        y_max = max(tb['box'][3] for tb in orphan_text_blocks)
        
        # Add some margin
        x_min = max(0, x_min - 20)
        y_min = max(0, y_min - 20)
        x_max = min(orphan_img.shape[1], x_max + 20)
        y_max = min(orphan_img.shape[0], y_max + 20)
        
        # Extract the region with orphan text
        orphan_region = orphan_img[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Process orphan content in columns
        if orphan_region.size > 0:
            orphan_columns = extract_text_by_columns(orphan_region, num_columns)
            
            # Process each column
            for i, column in enumerate(orphan_columns):
                if column['text']:
                    column['text'] = clean_ocr_text(column['text'])
                    column['text'] = enhance_text_with_llm(column['text'])
            
            # Create an article for orphan content
            orphan_content = " ".join([col['text'] for col in orphan_columns if col['text'].strip()])
            if orphan_content:
                article = {
                    'article_number': len(articles) + 1,
                    'headline': 'Untitled Article',
                    'subheadlines': [],
                    'columns': [],
                    'content': orphan_content,
                    'position': [x_min, y_min, x_max, y_max]
                }
                
                # Add column information
                for i, column in enumerate(orphan_columns):
                    if column['text'].strip():
                        article['columns'].append({
                            'column_number': i + 1,
                            'text': column['text']
                        })
                
                articles.append(article)
    
    # Sort articles by position (top to bottom)
    articles.sort(key=lambda x: x['position'][1])
    
    # Remove position data from final output
    for article in articles:
        del article['position']
    
    # Create the final JSON structure
    newspaper_data = {
        'source_image': os.path.basename(image_path),
        'total_articles': len(articles),
        'detected_columns': num_columns,
        'articles': articles
    }
    
    return newspaper_data

def process_directory(model, input_dir, output_dir, conf_threshold=0.05, ocr_languages=['en'], save_visualization=False, num_columns=None, auto_columns=False):
    """Process all images in a directory and output JSON files for each"""
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            newspaper_data = process_single_image(
                model, image_path, conf_threshold, ocr_languages, save_visualization, 
                None if auto_columns else num_columns  # Use None for auto-detection if auto_columns is True
            )
            
            if newspaper_data:
                # Save JSON output
                base_filename, _ = os.path.splitext(image_file)
                json_output_path = os.path.join(output_dir, f"{base_filename}.json")
                
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(newspaper_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved article structure to {json_output_path}")
            else:
                print(f"Failed to process image: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

def main():
    args = parse_args()
   
    # Parse and initialize OCR languages
    ocr_languages = args.languages.split(',')
    print(f"Using OCR languages: {ocr_languages}")
   
    # Check that either image or input_dir is specified
    if not args.image and not args.input_dir:
        print("Error: Either --image or --input_dir must be specified")
        return
   
    # Load model
    try:
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
   
    # Determine column detection method
    use_auto_columns = args.auto_columns
    specified_num_columns = args.num_columns
   
    if args.image:
        # Process a single image
        print(f"Processing image: {args.image}")
        newspaper_data = process_single_image(
            model, args.image, args.conf, ocr_languages,
            args.save_visualization,
            None if use_auto_columns else specified_num_columns  # Use None for auto-detection
        )
       
        if newspaper_data:
            # Create output directory if needed
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
           
            # Save JSON output
            base_filename = os.path.splitext(os.path.basename(args.image))[0]
            json_output_path = os.path.join(args.output_dir, f"{base_filename}.json")
           
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(newspaper_data, f, indent=2, ensure_ascii=False)
           
            print(f"Saved article structure to {json_output_path}")
            
            # Now call the cleaning process on the extracted data
            print(f"Cleaning extracted text with LLM...")
            json_str = json.dumps(newspaper_data)
            cleaned_article = process_newspaper_with_llm(json_str)
            
            # Save cleaned article to markdown file
            cleaned_output_path = os.path.join(args.output_dir, f"{base_filename}_cleaned.md")
            with open(cleaned_output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_article)
                
            print(f"Saved cleaned article to {cleaned_output_path}")
        else:
            print(f"Failed to process image: {args.image}")
   
    elif args.input_dir:
        # Process a directory of images
        print(f"Processing images in directory: {args.input_dir}")
        
        # Get a list of image files to process
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
            image_files.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext.upper()}")))
        
        for image_path in image_files:
            print(f"Processing image: {image_path}")
            newspaper_data = process_single_image(
                model, image_path, args.conf, ocr_languages,
                args.save_visualization,
                None if use_auto_columns else specified_num_columns
            )
            
            if newspaper_data:
                # Create output directory if needed
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                
                # Save JSON output
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                json_output_path = os.path.join(args.output_dir, f"{base_filename}.json")
                
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(newspaper_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved article structure to {json_output_path}")
                
                # Now call the cleaning process on the extracted data
                print(f"Cleaning extracted text with LLM...")
                json_str = json.dumps(newspaper_data)
                cleaned_article = process_newspaper_with_llm(json_str)
                
                # Save cleaned article to json file

                cleaned_output_path = os.path.join(args.output_dir, f"{base_filename}_cleaned.json")
                with open(cleaned_output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_article)
                    
                print(f"Saved cleaned article to {cleaned_output_path}")
                    
                
            else:
                print(f"Failed to process image: {image_path}")
   
    print("Text extraction and cleaning complete!")

if __name__ == "__main__":
    main()