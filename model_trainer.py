import re
import spacy
import random
from spacy.training import Example
import json

FILE_NAME = "labeled_jd.json"


def load_existing():
    with open(FILE_NAME, "r", encoding="utf_8")as f:
        return json.load(f)


def save_data(data):
    with open(FILE_NAME, "w", encoding="utf_8")as f:
        json.dump(data, f, indent=4)


dataset = []
raw_data = load_existing()

for item in raw_data:
    sent = item["text"]
    entities = []

    if "skills" in item:
        for skill in item["skills"]:
            if skill in sent:
                for match in re.finditer(re.escape(skill), sent):
                    entities.append((match.start(), match.end(), "SKILL"))

    if "languages" in item:
        for skill in item["languages"]:
            if skill in sent:
                for match in re.finditer(re.escape(skill), sent):
                    entities.append((match.start(), match.end(), "LANGUAGES"))

    dataset.append((sent, {"entities": entities}))

print(f"sucessfully loaded {len(dataset)} training example")
print(f"smple {dataset[0]}")


nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

ner.add_label("SKILL")
ner.add_label("LANGUAGE")

optimizer = nlp.begin_training()
EPOCHS = 40
for i in range(EPOCHS):
    random.shuffle(dataset)
    losses = {}  # track losses

    for text, annotations in dataset:
        doc = nlp.make_doc(text)

        example = Example.from_dict(doc, annotations)
        #  creating example object   dict  :   text -> annotations

        nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)

    print(f"for epoch {i}, we got {losses['ner']:.4f} losses")


nlp.to_disk("skill_extractor")
print("model saved...")
