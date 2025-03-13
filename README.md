# OCR Text Extraction and Processing System
##text to audio (voice.py) full code with eleven labs and voice cloing postive analysis 
## Overview

This document provides a detailed explanation of the OCR (Optical Character Recognition) system designed for newspaper content extraction and processing. The system consists of multiple components:

1. **Text Extraction**: Using EasyOCR to extract text from images
2. **Column Detection**: Analyzing layout to identify columns in newspaper pages
3. **Text Cleaning**: Processing OCR output with language models to fix errors
4. **Audio Conversion**: Converting cleaned text to audio using ElevenLabs API

## Text Extraction Components

### OCR Initialization

```python
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
```

### Command Line Arguments

```python
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
```

### Model Loading

```python
def load_model(model_path):
    """Load the trained YOLO model"""
    model = YOLO(model_path)
    return model
```

## Text Extraction Functions

### Region Text Extraction

```python
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
```

## Column Analysis and Processing

### Column-based Text Extraction

```python
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
```

### Column Layout Analysis

```python
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
```

### Article Processing by Columns

```python
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
```

## Notes

- The system uses YOLO object detection models to identify article elements
- EasyOCR is used for text extraction with multi-language support
- Column detection employs both image processing techniques and text distribution analysis
- Text enhancement is performed using language models to correct OCR errors
- Audio generation is handled by the ElevenLabs API (implementation details not shown)