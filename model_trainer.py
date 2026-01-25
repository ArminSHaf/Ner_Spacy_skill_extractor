import re
import spacy
import random
from spacy.training import Example
from spacy.util import minibatch, compounding
import json

TRAIN_FILE = "labeled_jd.json"
VAL_FILE = "validate.json"


def load_existing(file_name):
    with open(file_name, "r", encoding="utf_8")as f:
        raw_data = json.load(f)

    dataset = []
    for item in raw_data:
        sent = item["text"]
        entities = []

        if "skills" in item:
            for skill in item["skills"]:
                if skill.lower() in sent.lower():
                    pattern = r"\b" + re.escape(skill) + r"\b"
                    for match in re.finditer(pattern, sent, re.IGNORECASE):
                        entities.append((match.start(), match.end(), "SKILL"))

        if "languages" in item:
            for skill in item["languages"]:
                if skill.lower() in sent.lower():
                    pattern = r"\b" + re.escape(skill) + r"\b"
                    for match in re.finditer(pattern, sent, re.IGNORECASE):
                        entities.append(
                            (match.start(), match.end(), "LANGUAGES"))

        dataset.append((sent, {"entities": entities}))

    print(f"sucessfully loaded {len(dataset)} training example")
    print(f"smple {dataset[0]}\n")
    return dataset


train_data = load_existing(TRAIN_FILE)
val_data = load_existing(VAL_FILE)

#    spliting into different part traingin and validation for better training

# split_index = int(len(dataset) * 0.8)
# train_data = dataset[:split_index]
# val_data = dataset[split_index:]
# print(f"-📊.1  training model length : {len(train_data)}")
# print(f"-📊.2  val model length : {len(val_data)}\n")


nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

ner.add_label("SKILL")
ner.add_label("LANGUAGES")
optimizer = nlp.begin_training()

Max_EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 10
best_score = 0.0


best_loss = float('inf')
patience_counter = 0

print("Iteration | losses | f1(%) | precision | Recall | status")
for i in range(Max_EPOCHS):
    random.shuffle(train_data)
    losses = {}  # track losses

    batches = minibatch(train_data, size=BATCH_SIZE)

    for batch in batches:
        examples = []

        for text, annotations in batch:
            doc = nlp.make_doc(text)

            try:
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            except ValueError as e:
                print(f"⚠️ CRASH FOUND!")
                print(f"Sentence: '{text}'")
                print(f"Entities: {annotations['entities']}")
                print(f"Error: {e}")
                break  # Stop so you can read it
        if examples:
            nlp.update(examples, drop=0.4, sgd=optimizer, losses=losses)

    #  VALIDATION
    val_examples = []
    for text, annotations in val_data:
        doc = nlp.make_doc(text)
        val_examples.append(Example.from_dict(doc, annotations))

    scores = nlp.evaluate(val_examples)  # evaluate function
    current_f1 = scores['ents_f']
    current_prec = scores['ents_p']
    current_rec = scores['ents_r']

    status = ""
    if current_f1 > best_score:
        best_score = current_f1
        patience_counter = 0
        nlp.to_disk("skill_extractor")
        status = "💾💾💾model saved..."
    else:
        patience_counter += 1
        status = f"   ----- not improved  {patience_counter} / {PATIENCE}"

    print(
        f"{i+1:<8} | {losses['ner']:<8.2f} | {current_f1*100:<8.1f} | {current_prec*100:<10.1f} | {current_rec*100:<8.1f} | {status}")

    if patience_counter == PATIENCE:
        print(
            f"\n🛑 Early stopping triggered. Best F1 Score: {best_score*100:.2f}%")

        break

print("training completed and saved")
