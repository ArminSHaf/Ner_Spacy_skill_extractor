# Ner Spacy skill extractor

This project explores building and fine-tuning a spaCy Named Entity Recognition (NER) model to extract **skills**  (technical and also non-technical) from text such as programming languages, tools, and frameworks.

## setup
Load the spacy model using the command bellow at your terminal

```python
pip install spacy
```


## model verification

At this stage, the project uses the pretrained `en_core_web_sm` model to analyze simple sentences.
after verification It is quite observable that the model is not edaquately trained to extract skills from sentences.

