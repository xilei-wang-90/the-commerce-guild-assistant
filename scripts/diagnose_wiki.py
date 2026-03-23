import sys
import re
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

class VerboseConverter(MarkdownConverter):
    def process_tag(self, node, *args, **kwargs):
        tag_name = getattr(node, 'name', 'text_node')
        if tag_name and tag_name != '[document]':
            print(f"DEBUG [ENTER]: <{tag_name}>")
        
        result = super().process_tag(node, *args, **kwargs)
        
        if tag_name and tag_name != '[document]':
            preview = result.strip().replace('\n', '\\n')[:50]
            print(f"DEBUG [EXIT]: </{tag_name}> -> Result: '{preview}...'")
        return result

def surgical_clean(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # --- PASTE YOUR WORKER CLEANING LOGIC HERE ---
    # Example (match your Worker exactly):
    # for junk in soup.select('.toc, .stub'): junk.decompose()
    # promote_first_row_to_header(soup)
    # etc.
    # ---------------------------------------------
    
    return soup

def diagnose_file(file_path):
    print(f"--- Starting REAL Diagnosis for: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_raw = f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Step 1: Manipulate the soup just like your Worker does
    clean_soup = surgical_clean(html_raw)
    
    # Step 2: Convert the manipulated soup string
    converter = VerboseConverter(
        heading_style="ATX",
        strip=['a', 'span', 'script', 'style']
    )
    
    # Convert the string version of the clean soup
    final_md = converter.convert(str(clean_soup))
    
    print("\n--- Diagnosis Complete ---")
    output_path = "diagnostic_result.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_md)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_wiki.py path_to_your_file.html")
    else:
        diagnose_file(sys.argv[1])
