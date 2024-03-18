from natasha import (
    Segmenter,
    MorphVocab,
    PER,
    NewsNERTagger,
    NewsEmbedding,
    Doc)
import spacy
from sklearn.metrics import f1_score
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
emb = NewsEmbedding()
segmenter = Segmenter()
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()


def extract_names_spacy(text, epochs):
    nlp = spacy.load(f"technology_test_models/ru_core_news_md_{epochs}")  # Загрузка модели для извлечения именованных сущностей
    doc = nlp(text)
    names = []
    for ent in doc.ents:
        if ent.label_ == "TECHNOLOGY":
            name = ent.text.split()
            names += name
    return names


# Текстовые данные для извлечения именованных сущностей ФИО
with open('test_set_technologies_ner.txt', 'r') as file:
    text = file.read()

# Золотой стандарт (правильные ФИО)
golden_standard = ['excel', 'python', 'c++', 'word', 'html', 'css', 'java', 'django', 'php']

# Извлечение ФИО с помощью Natasha и SpaCy


for i in range(26):
    spacy_names = extract_names_spacy(text, i)
    spacy_extracted = spacy_names

    spacy_set = set(spacy_names)
    golden_set = set(golden_standard)

    # Рассчитываем TP (True Positives), FP (False Positives) и FN (False Negatives)
    spacy_tp = len(spacy_set.intersection(golden_set))
    spacy_fp = len(spacy_set.difference(golden_set))
    spacy_fn = len(golden_set.difference(spacy_set))
    # Рассчитываем Precision, Recall и F1-score
    spacy_precision = spacy_tp / (spacy_tp + spacy_fp) if (spacy_tp + spacy_fp) > 0 else 0
    spacy_recall = spacy_tp / (spacy_tp + spacy_fn) if (spacy_tp + spacy_fn) > 0 else 0
    spacy_f1 = 2 * (spacy_precision * spacy_recall) / (spacy_precision + spacy_recall) if (spacy_precision + spacy_recall) > 0 else 0
    print(f"epochs: {i}  F1-score: {spacy_f1:.2f}  Precision: {spacy_precision:.2f}  Recall: {spacy_recall:.2f}")

