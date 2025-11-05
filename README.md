# Detection of Missing Information & Vague Pronoun Detector

This project helps find problems in Software Requirements Specifications (SRS) like missing info or unclear references.

## Files

- `extracting.py`  
  Creates a dataset of sentences with missing info indicators like "TBD", "to be decided", or blanks.

- `model.py`  
  Trains a Transformer model to detect problematic sentences in requirements.

- `testing.py`  
  Tests your model on an input PDF and produces an annotated PDF highlighting issues with recommendations.

- External dataset "Definite Pronoun Resolution Dataset" is used to detect unclear or ambiguous pronouns (like "it", "they") in the text.

## Features

- Automated detection of missing or vague information with common indicators like TBD.
- Pronoun detection and basic anaphora resolution to find ambiguous references.
- Annotated PDF output with sentence highlights and inline recommendations.

## How to use

1. Run `extracting.py` to build your dataset.  
2. Train the model with `model.py`.  
3. Run `testing.py` with your PDF to get an annotated output.

