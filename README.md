# Detection of Missing Information & Vague Pronoun Detector

This project aims to automatically identify problematic sentences in Software Requirements Specifications (SRS) related to missing information and ambiguous pronoun references, thereby improving document clarity and review efficiency.

## Project Overview

- `Dataset Creation`  
  A synthetic dataset was created comprising sentences that exhibit indicators of missing information such as "TBD", "to be decided", blank values.

- `Model Training`  
  Two Transformer-based models were trained independently:  
- One model was trained on the synthetic missing information dataset to detect vague specifications.  
- The other model was trained on the Definite pronoun resolution dataset to recognize ambiguous pronoun references.

- `Testing and Annotation`  
  The system processes SRS documents by extracting text, segmenting it into sentences and evaluating each sentence independently through both models. Sentences flagged by either model are annotated with highlights and context-specific recommendations to aid in clarifying vague terms and ambiguous pronouns. 


## How to use

1. Run `extracting.py` to build your dataset.  
2. Train each model independently using its respective dataset.  
3. Run the system on SRS documents to obtain annotated outputs with flagged sentences and recommendations.

**Note:** All files required for training and testing both models have been uploaded.

