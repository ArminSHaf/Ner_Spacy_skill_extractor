import re
import spacy
from spacy.training import Example


with open("jd.txt", "r", encoding="utf_8")as f:
    jd_lines = f.readlines()

my_trained_model = spacy.load("skill_extractor")

test_sentence = "Communication: Strong verbal and written communication skills, with the ability to convey complex technical concepts to non-technical stakeholders. "


print("\n result : ")
for j in jd_lines:
    if not j.strip():
        continue
    print("\n for sentence :", j)
    doc = my_trained_model(j)
    for ent in doc.ents:
        print("text ", ent.text, " -   label : ", ent.label_)
