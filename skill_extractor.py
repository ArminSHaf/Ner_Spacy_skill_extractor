import spacy

nlp = spacy.load('en_core_web_sm')

texts = [
    "I love programming",
    "you must know Java",
    "Required skills : Python, GOLang, and other tools",
    "JavaScript"
]

# ner = nlp.get_pipe('ner').labels
# print(ner)


docs = [nlp(text) for text in texts]
for doc in docs:
    entities = []
    for ent in doc.ents:
        entities.append([ent.text, ent.label_])

    print(entities)
