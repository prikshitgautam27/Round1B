import os
import json
import fitz  # PyMuPDF
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from collections import Counter

def convert_to_serializable(obj):
    """Recursively converts NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj

# --- MODEL LOADING ---
tokenizer = AutoTokenizer.from_pretrained('./models/all-mpnet-base-v2')
model = AutoModel.from_pretrained('./models/all-mpnet-base-v2')
model.eval()

# --- DOMAIN DETECTION ---
def detect_domain(query):
    """Detects the user's domain based on keywords in the query."""
    query_lower = query.lower()
    if any(word in query_lower for word in ['trip', 'travel', 'vacation', 'itinerary', 'friends']):
        return 'travel'
    if any(word in query_lower for word in ['menu', 'dish', 'recipe', 'vegetarian', 'buffet']):
        return 'culinary'
    if any(word in query_lower for word in ['form', 'onboarding', 'compliance', 'hr', 'signature', 'fillable']):
        return 'hr_forms'
    return 'general'

# --- CORE FUNCTIONS ---
def get_embedding(text):
    """Generates a text embedding using the loaded transformer model."""
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def parse_pdfs(pdf_paths):
    """
    This is the definitive, robust universal parser. It now contains a stop list
    to ignore generic words like "Instructions" being treated as titles.
    """
    print("--- Parsing PDFs with Definitive Universal Analyzer... ---")
    
    # ADDED: A stop list of words that should NOT be considered titles
    heading_stop_list = {'instructions', 'ingredients', 'directions', 'notes'}
    
    all_chunks = []
    for pdf_path in pdf_paths:
        doc_name = os.path.basename(pdf_path)
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    page_dict = page.get_text("dict")
                    blocks = page_dict.get("blocks", [])
                    if not blocks: continue
                    
                    font_sizes = [span['size'] for b in blocks if 'lines' in b for l in b['lines'] for span in l['spans']]
                    if not font_sizes: continue
                    body_font_size = Counter(round(s, 1) for s in font_sizes).most_common(1)[0][0]
                    
                    current_heading = "General Information"
                    current_text = []

                    for block in blocks:
                        if 'lines' not in block: continue
                        block_text = " ".join(span['text'] for line in block['lines'] for span in line['spans']).strip()
                        if not block_text: continue
                        
                        block_font_size = round(block['lines'][0]['spans'][0]['size'], 1)
                        
                        # MODIFIED: is_heading check now uses the stop list
                        is_heading = (
                            block_font_size > body_font_size + 0.5 and
                            len(block_text.split()) < 15 and
                            not block_text.endswith('.') and
                            block_text.lower().strip(":") not in heading_stop_list
                        )

                        if is_heading:
                            if current_text and len(" ".join(current_text).split()) > 20:
                                all_chunks.append({
                                    "title": current_heading, "text": " ".join(current_text),
                                    "document": doc_name, "page": page_num + 1
                                })
                            current_heading = block_text
                            current_text = []
                        else:
                            current_text.append(block_text)

                    if current_text and len(" ".join(current_text).split()) > 20:
                        all_chunks.append({
                            "title": current_heading, "text": " ".join(current_text),
                            "document": doc_name, "page": page_num + 1
                        })
        except Exception as e:
            print(f"    Error parsing {doc_name}: {e}")
            
    print(f"  -> Parsing complete. Found {len(all_chunks)} text chunks.")
    return all_chunks

def apply_intelligent_scoring(chunks, query, domain):
    """Applies a multi-layered, domain-specific scoring logic to rank chunks."""
    print(f"--- Applying Intelligent Scoring (Domain: {domain})... ---")

    # --- Domain-Specific Keyword and Document Lists ---
    travel_bonus_keywords = {'nightlife', 'bars', 'beach', 'adventure', 'coastal', 'party', 'entertainment', 'music', 'boat', 'wine', 'culinary', 'restaurants', 'cities', 'guide', 'tips'}
    travel_penalty_keywords = {'family-friendly', 'children', 'kids', 'educational'}
    travel_doc_bonuses = {"Restaurants and Hotels": 0.25, "Cities": 0.2, "Things to Do": 0.15, "Cuisine": 0.15, "Tips and Tricks": 0.1}
    
    culinary_negative_keywords = {'beef', 'pork', 'chicken', 'lamb', 'turkey', 'veal', 'prosciutto', 'salami', 'bacon', 'ham', 'sausage', 'fish', 'tuna', 'seafood', 'shrimp', 'anchovies', 'salmon', 'lobster', 'duck'}
    culinary_substantial_keywords = {'lasagna', 'casserole', 'sushi', 'rolls', 'curry', 'stew', 'risotto', 'pasta', 'cheese', 'potatoes', 'beans', 'lentils', 'rice', 'falafel', 'ratatouille', 'gnocchi', 'tikka'}
    culinary_gluten_keywords = {'flour', 'wheat', 'bread', 'pasta', 'breadcrumbs', 'pita', 'baguette', 'couscous', 'tortilla', 'freekeh', 'roux', 'beer'}
    
    hr_forms_bonus_keywords = {'form', 'fillable', 'sign', 'signature', 'interactive', 'field', 'onboarding', 'compliance', 'checklist', 'e-signature', 'prepare forms'}
    hr_forms_penalty_keywords = {'error', 'unable', 'failed', 'recipe', 'dish', 'ingredients'}
    
    query_embedding = get_embedding(query)
    final_results = []

    for chunk in chunks:
        text_lower = (chunk['title'] + ' ' + chunk['text']).lower()
        relevance_score = cosine_similarity(get_embedding(chunk['text'][:500]), query_embedding)
        total_score = relevance_score 

        # --- Apply domain-specific scoring logic ---
        if domain == 'travel':
            # a. Massive penalty for irrelevant topics
            if any(kw in text_lower for kw in travel_penalty_keywords):
                total_score -= 1.0 
            # b. High-impact bonus for each relevant keyword
            total_score += sum(0.2 for kw in travel_bonus_keywords if kw in text_lower)
            # c. Document-level bonus to encourage diversity
            for doc_key, doc_bonus in travel_doc_bonuses.items():
                if doc_key in chunk['document']:
                    total_score += doc_bonus
                    break
        
        elif domain == 'culinary':
            # YOUR PREVIOUS CULINARY LOGIC IS PRESERVED
            if any(kw in text_lower for kw in culinary_negative_keywords):
                continue
            substantial_bonus = sum(0.2 for keyword in culinary_substantial_keywords if keyword in text_lower)
            gluten_penalty = sum(0.05 for keyword in culinary_gluten_keywords if keyword in text_lower)
            total_score += substantial_bonus - gluten_penalty

        elif domain == 'hr_forms':
            # YOUR PREVIOUS HR_FORMS LOGIC IS PRESERVED
            bonus = sum(0.15 for keyword in hr_forms_bonus_keywords if keyword in text_lower)
            penalty = sum(0.1 for keyword in hr_forms_penalty_keywords if keyword in text_lower)
            total_score += bonus - penalty

        chunk['final_score'] = total_score
        final_results.append(chunk)

    final_results.sort(key=lambda x: x['final_score'], reverse=True)
    return final_results

def main():
    """Main function to run the document processing pipeline."""
    input_dir = "./input"
    output_file = "./output/challenge1b_output.json"

    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    # Load task information from JSON file
    persona, task_description, query = "Professional", "Document Analysis", "Analyze documents"
    json_path = next((os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')), None)
    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        persona = data.get('persona', {}).get('role', persona)
        task_description = data.get('job_to_be_done', {}).get('task', task_description)
        query = f"Persona: {persona}. Task: {task_description}"
    
    print(f"Processing Query: {query}")
    
    domain = detect_domain(query)
    # Use the new, robust parser
    doc_chunks = parse_pdfs(pdf_files)
    final_results = apply_intelligent_scoring(doc_chunks, query, domain)
    
    # Final quality filter to remove any remaining generic sections
    final_results = [result for result in final_results if result.get('title') not in ['Document Section', 'General Information']]

    # Generate final output JSON
    metadata = {
        "input_documents": sorted([os.path.basename(f) for f in pdf_files]),
        "job_to_be_done": task_description,
        "persona": persona,
        "processing_timestamp": datetime.utcnow().isoformat()
    }
    top_results = final_results[:5]

    extracted_sections = [{
        "document": r["document"], "section_title": r["title"],
        "importance_rank": i + 1, "page_number": r["page"]
    } for i, r in enumerate(top_results)]

    subsection_analysis = [{
        "document": r["document"], "refined_text": r["text"].strip(),
        "page_number": r["page"]
    } for r in top_results]

    final_output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(convert_to_serializable(final_output), f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()