import xml.etree.ElementTree as ET
import re
import os

def extract_text(file_path: str) -> str:
    """
    Extracts text content from an earnings transcript XML file.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract all text content from XML elements, including CDATA
    text_parts = []
    
    # Look specifically for Body and Headline elements which contain CDATA
    for event_story in root.findall('.//EventStory'):
        # Extract headline
        headline = event_story.find('Headline')
        if headline is not None and headline.text:
            text_parts.append(headline.text.strip())
        
        # Extract body
        body = event_story.find('Body')
        if body is not None and body.text:
            text_parts.append(body.text.strip())
    
    # Join all text parts with spaces
    raw_text = ' '.join(text_parts)
    
    # Clean up the text
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    # Remove special characters and normalize whitespace
    cleaned_text = re.sub(r'[^\w\s.,?!-]', '', cleaned_text)
    # Trim leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text



if __name__ == "__main__":
    file_path = input("Enter the path to the XML file: ")
    
    try:
        text = extract_text(file_path)
        if text:
            print("\nExtracted text (first 500 characters):")
            print(text[:500], "...\n")
            print(f"Total characters extracted: {len(text)}")
        else:
            print("\nNo text was extracted from the XML file")
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
    except ET.ParseError:
        print("Error: Invalid XML file")

print(type(text))
print(text)