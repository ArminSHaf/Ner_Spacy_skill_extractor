import re
import spacy
import random
from spacy.training import Example


texts = [
    "I love programming",
    "you must know Java",
    "Required skills : Python, GOLang, and other tools",
    "JavaScript"
]

# ner = nlp.get_pipe('ner').labels
# print(ner)


# docs = [nlp(text) for text in texts]
# for doc in docs:
#     entities = []
#     for ent in doc.ents:
#         entities.append([ent.text, ent.label_])

#     print(entities)


def create_training_data(sentence, skills):
    entities = []
    for skill in skills:
        if skill in sentence:
            # escape is basically for special char to not throw an error
            for match in re.finditer(re.escape(skill), sentence):
                entities.append((match.start(), match.end(), "SKILL"))

    return (sentence, {"entities": entities})


skill_list = ["Python", "Java", "C++", "SQL", "Git", "React", "Docker", "AWS"]

raw_sentences = [
    "We are looking for a Python developer with 3 years of experience.",
    "Strong knowledge of Java and C++ is required.",
    "Experience with SQL databases and Git version control.",
    "React frontend experience is a plus.",
    "Must know Docker and AWS for cloud deployment.",
    "Python is great, but we also use Java.",
    "We need a ninja in SQL and Python.",
]
dataset = []
for sent in raw_sentences:
    data = create_training_data(sent, skill_list)

    dataset.append(data)
print("created dataset with :", len(dataset))


nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

optimizer = nlp.begin_training()
EPOCHS = 20
for i in range(EPOCHS):
    random.shuffle(dataset)
    losses = {}  # track losses

    for text, annotations in dataset:
        doc = nlp.make_doc(text)

        example = Example.from_dict(doc, annotations)
        #  creating example object   dict  :   text -> annotations

        nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)

    print(f"for epoch {EPOCHS}, we got {losses}")


nlp.to_disk("skill_extractor")
print("model saved...")


my_trained_model = spacy.load("skill_extractor")

test_sentence = " you are required to know : Java, BDE and other tools"

doc = my_trained_model(test_sentence)

print("\n result : ")
for ent in doc.ents:
    print("text ", ent.text, " -   label : ", ent.label_)
