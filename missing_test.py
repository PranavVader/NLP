import os
import fitz  
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

model_path = "fine_tuned_model5"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
nlp = spacy.load("en_core_web_sm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

folder_path = r"C:/Users/asust/Downloads/1414117/requirements/req"

missing_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        raw_text = extract_text_from_pdf(pdf_path)
        doc = nlp(raw_text)
        sentences = [sent.text.strip() for sent in doc.sents]

        ambiguous_count = 0
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=256, padding='max_length').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_label = outputs.logits.argmax(dim=1).item()

            if predicted_label == 1:
                ambiguous_count += 1

        print(f"File: {filename} - missing sentences: {ambiguous_count}")
        missing_count += ambiguous_count

print(f"\nTotal ambiguous sentences across all files: {missing_count}")


import fitz  
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def simple_recommendation(sentence):
    keywords = ["TBD", "to be decided", "pending", "unknown", "blank", "not specified"]
    sentence_lower = sentence.lower()
    for kw in keywords:
        if kw.lower() in sentence_lower:
            return f"Replace vague term '{kw}' with specific details."
    
    return "Review for clarity and completeness."

def annotate_pdf_with_recommendations(pdf_path, output_path, flagged_sentences):
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        for sentence in flagged_sentences:
            areas = page.search_for(sentence)
            for rect in areas:
                highlight = page.add_highlight_annot(rect)
                recommend_text = simple_recommendation(sentence)
                page.add_text_annot(rect.tl, recommend_text)
                highlight.update()
    doc.save(output_path)
    print(f"Saved annotated PDF to {output_path}")

model_path = "fine_tuned_model5"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    pdf_path = "C:/Users/asust/Downloads/1414117/requirements/req/1998 - themas.pdf"
    output_pdf_path = "C:/Users/asust/Downloads/1414117/requirements/req/trial5.pdf"

    raw_text = extract_text_from_pdf(pdf_path)
    doc = nlp(raw_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    flagged_sentences = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = outputs.logits.argmax(dim=1).item()

        if predicted_label == 1:
            flagged_sentences.append(sentence)

    annotate_pdf_with_recommendations(pdf_path, output_pdf_path, flagged_sentences)
