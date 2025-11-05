import fitz  
import os
import random
import csv

missing_markers = [
    "TBD",
    "to be decided",
    "",
    "N/A",
]


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def insert_missing_info(sentence):
    words = sentence.split()
    if len(words) < 4:
        return sentence, 0  
    
    idx = random.randint(1, len(words)-2) 
    replacement = random.choice(missing_markers)
    words[idx] = replacement
    modified_sentence = " ".join(words)
    return modified_sentence, 1

def process_srs_pdfs(pdf_folder, output_file):
    dataset = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Extracting from: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        sentences = text.split('.')
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            
            if random.random() > 0.5:
                mod_sent, label = insert_missing_info(sent)
            else:
                mod_sent, label = sent, 0
            
            dataset.append((mod_sent, label))
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sentence', 'missing_info_label'])
        writer.writerows(dataset)
    print(f"Dataset saved to {output_file}")

pdf_folder = r"C:/Users/asust/Downloads/1414117/requirements/req"
output_file = "synthetic_missing_info_dataset.csv"
process_srs_pdfs(pdf_folder, output_file)
