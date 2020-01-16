"""

IDE: PyCharm
Project: complete-sentence-prediction
Author: Robin
Filename: prepare_data.py
Date: 16.01.2020

"""
import csv
from os import listdir
from random import shuffle

import spacy
from os.path import isfile, join
from tqdm import tqdm

from paths import EXPRESS_DATA_DIR, QUESTION_DATA_DIR, DATA_DIR

# define parameters
CORRECT_LABEL = 1
INCORRECT_LABEL = 0
min_tokens = 5
dataset = []
nlp = spacy.load("en_core_web_sm")

# process expression docs
expressions_docs = [join(EXPRESS_DATA_DIR, f) for f in listdir(EXPRESS_DATA_DIR) if isfile(join(EXPRESS_DATA_DIR, f))]

for edoc in tqdm(expressions_docs):
    with open(edoc, "r", encoding="utf-8") as edoc_file:
        start_line = 10
        for index, line in enumerate(edoc_file):
            if ":" in line or index < start_line:
                continue
            else:
                sent = line.strip().replace("\n", "")
                if not line.strip() == "":
                    tokens = [token.text for token in nlp(sent)][:-1]
                    if len(tokens) >= min_tokens:
                        dataset.append([tokens, CORRECT_LABEL])

# process question docs
questions_docs = [join(QUESTION_DATA_DIR, f) for f in listdir(QUESTION_DATA_DIR) if isfile(join(QUESTION_DATA_DIR, f))]

for edoc in tqdm(questions_docs):
    with open(edoc, "r", encoding="utf-8") as edoc_file:
        for line in edoc_file:
            if not line.strip() == "":
                analyzed_line = nlp(line.strip().replace("\n", ""))
                for sent in analyzed_line.sents:
                    tokens = [token.text for token in sent][:-1]
                    if len(tokens) >= min_tokens:
                        dataset.append([tokens, CORRECT_LABEL])

# randomize order
shuffle(dataset)


def create_wrong_sample(entry):
    """
    Modifies tokens in a sentences so that it becomes incomplete
    :param entry:
    :return:
    """
    tokens = entry[0][:-2]
    return [tokens, INCORRECT_LABEL]


# adding equally wrong examples
pbar = tqdm(total=len(dataset))
for i in range(len(dataset) + 0):
    entry = dataset[i]
    wrong_entry = create_wrong_sample(entry)
    dataset.append(wrong_entry)
    pbar.update(1)
pbar.close()

# shuffle again
shuffle(dataset)

# split into train/test (80/20)
split = int(len(dataset) * 0.8)
train, test = dataset[:split], dataset[split:]


def save_as_csv(filename, entries):
    with open(DATA_DIR + "/" + filename, "w+", encoding="utf-8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for line in tqdm(entries):
            csv_writer.writerow([' '.join(line[0]).strip(), line[1]])


save_as_csv("generated_train.csv", train)
save_as_csv("generated_test.csv", test)
