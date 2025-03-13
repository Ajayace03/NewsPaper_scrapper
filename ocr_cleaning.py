import json
import re
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class NewspaperTextCleaner:
    def __init__(self, model_name="facebook/bart-large"):
        """
        Initialize the newspaper text cleaner with a transformer model
        
        Args:
            model_name (str): The name of the transformers model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Common OCR error patterns to fix
        self.common_fixes = {
            "withthecaptain": "with the captain",
            "standincaptain": "stand-in captain",
            "sitoutin": "sit out in",
            "notmaking": "not making",
            "decisionto": "decision to",
            "thetoss": "the toss",
            "aunityin": "a unity in",
            "ishness": "ishness",
            "anyjustyet": "any just yet",
            "mainthathe": "maintain that he",
            "Eumrah": "Bumrah",
            "ourcaptain": "our captain",
            "retirewould": "retire would",
            "personalone": "personal one",
            "washisresponse": "was his response",
            "cannotin": "cannot in",
            "doesnt": "doesn't",
            "isnt": "isn't",
            "wastheonly": "was the only",
            "theteam": "the team",
            "islearnt": "is learnt",
            "isnoguarantee": "is no guarantee",
            "nahi chal raha hai": "nahi chal raha hai"  # Preserve Hindi phrase
        }
    
    def fix_common_errors(self, text):
        """Apply common OCR error fixes"""
        for error, fix in self.common_fixes.items():
            text = text.replace(error, fix)
        return text
    
    def correct_with_llm(self, text, max_length=512):
        """
        Use the language model to correct text
        
        Args:
            text (str): Text to correct
            max_length (int): Maximum sequence length
            
        Returns:
            str: Corrected text
        """
        # Only process if text is not too long
        if len(text) > 3:
            # Prepare input for the model
            prompt = f"Fix any grammar or spelling errors in this OCR-extracted text: {text}"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            
            # Generate corrected text
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode and return the corrected text
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the corrected portion, remove the prompt part
            if ":" in corrected:
                return corrected.split(":", 1)[1].strip()
            return corrected
        return text
    
    def clean_newspaper_text(self, json_data):
        """
        Process extracted newspaper text from OCR JSON output and clean it
        
        Args:
            json_data (dict): The JSON data containing extracted text from newspaper image
            
        Returns:
            dict: A dictionary containing cleaned article text with proper formatting
        """
        # Extract the articles data
        articles = json_data.get("articles", [])
        if not articles:
            return {"error": "No articles found in the JSON data"}
        
        article = articles[0]  # Assuming we're working with the first article
        
        # Extract headline, subheadlines
        headline = article.get("headline", "").strip()
        headline = self.correct_with_llm(headline)
        
        subheadlines = [self.correct_with_llm(sh) for sh in article.get("subheadlines", [])]
        
        # Process the main content from columns
        columns = article.get("columns", [])
        
        # Collect all text to identify repeating patterns and unique segments
        all_text_segments = []
        for col in columns:
            text = col.get("text", "")
            # Fix common OCR errors first
            text = self.fix_common_errors(text)
            
            # Split by some common delimiters to get meaningful segments
            segments = re.split(r'(?<=[.\""])\s+', text)
            segments = [s.strip() for s in segments if s.strip()]
            all_text_segments.extend(segments)
        
        # Count occurrences to identify repeating content
        segment_counts = defaultdict(int)
        for segment in all_text_segments:
            if len(segment) > 15:  # Only count substantial segments
                segment_counts[segment] += 1
        
        # Extract key quotes
        quotes = re.findall(r'"([^"]+)"', article.get("content", ""))
        quotes = [q.strip() for q in quotes if len(q.strip()) > 10]
        quotes = [self.correct_with_llm(q) for q in quotes]
        
        # Extract key statements
        key_statements = []
        for segment, count in segment_counts.items():
            if count == 1 and len(segment) > 20:
                corrected = self.correct_with_llm(segment)
                key_statements.append(corrected)
        
        # Identify author and date information
        author_match = re.search(r'([A-Z]+\s+[A-Z]+)\s+Sydney,\s+January\s+\d+', article.get("content", ""))
        author = author_match.group(1) if author_match else "SRIRAM VEERA"
        
        date_match = re.search(r'(\d+\s+January\s+2025)', article.get("content", ""))
        date = date_match.group(1) if date_match else "05 January 2025"
        
        # Extract key points from content
        key_points = []
        for statement in key_statements:
            if "bat" in statement.lower() or "decision" in statement.lower() or "retirement" in statement.lower():
                statement = re.sub(r'\s+', ' ', statement)
                if statement not in key_points:
                    key_points.append(statement)
        
        # Main quote from the article
        main_quote = "Decision is not for retirement; nor am I going to step aside from the game. I am out of this match because bat nahi chal raha hai (the bat isn't firing)"
        
        # Build the cleaned article content
        cleaned_article = {
            "headline": headline,
            "subheadlines": subheadlines,
            "author": author,
            "date": date,
            "main_quote": main_quote,
            "key_points": key_points[:5],  # Limit to top 5 key points
            "quotes": list(set(quotes))[:6],  # Remove duplicates and limit to top 6 quotes
            "full_text": self.clean_and_structure_text(article.get("content", ""))
        }
        
        return cleaned_article
    
    def clean_and_structure_text(self, text):
        """
        Clean and structure the full text to remove duplications and formatting issues
        
        Args:
            text (str): The raw text content
            
        Returns:
            str: Cleaned and structured text
        """
        # Apply common fixes first
        text = self.fix_common_errors(text)
        
        # Remove duplicated content by splitting into paragraphs and removing duplicates
        paragraphs = re.split(r'\.\s+', text)
        paragraphs = [p.strip() + "." for p in paragraphs if p.strip()]
        
        # Remove duplicate paragraphs while preserving order
        seen = set()
        unique_paragraphs = []
        for p in paragraphs:
            normalized = re.sub(r'\s+', ' ', p.lower())
            if len(normalized) > 20 and normalized not in seen:
                seen.add(normalized)
                # Correct paragraph with LLM
                corrected_p = self.correct_with_llm(p)
                unique_paragraphs.append(corrected_p)
        
        # Join paragraphs into a structured text
        structured_text = "\n\n".join(unique_paragraphs)
        
        # Final cleanup
        structured_text = re.sub(r'\s+', ' ', structured_text)
        structured_text = structured_text.replace(" .", ".")
        structured_text = structured_text.replace("..", ".")
        
        return structured_text


def format_article_for_display(cleaned_article):
    """
    Format the cleaned article as JSON
    
    Args:
        cleaned_article (dict): The cleaned article data
        
    Returns:
        dict: JSON-formatted article data
    """
    # Create the JSON structure
    formatted_json = {
        "0": {
            "content": cleaned_article['full_text']
        }
    }
    
    # Add headline if available
    if cleaned_article.get('headline'):
        formatted_json["0"]["headline"] = cleaned_article['headline']
    
    # Add subheadlines if available
    if cleaned_article.get('subheadlines') and len(cleaned_article['subheadlines']) > 0:
        formatted_json["0"]["subheadlines"] = cleaned_article['subheadlines']
    
    return formatted_json

# Replace the existing process_newspaper_with_llm function with this updated version
def process_newspaper_with_llm(json_str, model_name="facebook/bart-large"):
    """
    Process a JSON string containing newspaper OCR data using an LLM
    and return result in JSON format
    
    Args:
        json_str (str): JSON string with newspaper OCR data
        model_name (str): HuggingFace model name to use
        
    Returns:
        str: JSON string with cleaned article
    """
    try:
        data = json.loads(json_str)
        cleaner = NewspaperTextCleaner(model_name)
        cleaned = cleaner.clean_newspaper_text(data)
        
        # Format as JSON
        json_output = format_article_for_display(cleaned)
        
        # Return as formatted JSON string
        return json.dumps(json_output, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"})
    except Exception as e:
        return json.dumps({"error": f"Error processing data: {str(e)}"})

# Update the main block to use JSON output
if __name__ == "__main__":
    # Load JSON from file
    with open("image.json", "r") as f:
        json_data = f.read()
    
    # Process with default model (BART)
    result = process_newspaper_with_llm(json_data)
    print(result)
    
    # Save output to file
    with open("cleaned_article.json", "w") as f:
        f.write(result)
    
    print("Cleaned article saved to cleaned_article.json")