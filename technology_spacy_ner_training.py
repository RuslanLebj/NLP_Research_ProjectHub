import spacy
from spacy.training import Example
import json
import random

nlp = spacy.load("ru_core_news_md")

# Загрузка и подготовка данных из JSONL
with open('datasets/technology_ner/technologies.jsonl', 'r', encoding='utf-8') as file:
    training_data = [json.loads(line) for line in file]

# Подготовка данных для дообучения
TRAIN_DATA = []
for item in training_data:
    words = item.get('words')
    labels = item.get('labels')
    entities = [(0, len(words[0]), labels[0])]  # Предполагаем, что каждое слово - это одна сущность

    TRAIN_DATA.append((words[0], {'entities': entities}))

# Добавление новых энтити-меток к модели
ner = nlp.get_pipe("ner")
for label in set([label for _, annotations in TRAIN_DATA for label in annotations.get('entities')[0][2]]):
    ner.add_label(label)

# Дообучение модели
epochs = 26  # количество эпох обучения
for _ in range(epochs):
    print(_)
    random.shuffle(TRAIN_DATA)
    losses = {}
    examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    nlp.update(examples, drop=0.5, losses=losses)

    nlp.to_disk(f"technology_test_models/ru_core_news_md_{_}")

