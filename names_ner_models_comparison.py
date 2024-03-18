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


# Функции извлечения ФИО с помощью Natasha и SpaCy
def extract_names_natasha(text):
    names = []
    doc = Doc(text)
    # Деление на предложения и токены
    doc.segment(segmenter)
    # Отмечаем части предложения
    doc.tag_ner(ner_tagger)
    # Нормализуем
    for span in doc.spans:
        span.normalize(morph_vocab)
    # Выделяем все имена и добавляем в список
    for span in doc.spans:
        if span.type == PER:
            names += span.text.split(" ")
    return names

def extract_names_spacy(text):
    nlp = spacy.load("ru_core_news_sm")  # Загрузка модели для извлечения именованных сущностей
    doc = nlp(text)
    names = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            name = ent.text.split()
            names += name
    return names


# Текстовые данные для извлечения именованных сущностей ФИО
with open('test_set_names_ner.txt', 'r') as file:
    text = file.read()

# Золотой стандарт (правильные ФИО)
golden_standard = ['Малыш', 'Михаил', 'Сергеевич', 'Омаров', 'Руслан', 'Теймурович', 'Плахота', 'Владислав',
                   'Александрович', 'Чеканова', 'Елизавета', 'Дмитриевна', 'Подольная' 'Светлана', 'Сергеевна' ]

# Извлечение ФИО с помощью Natasha и SpaCy
natasha_names = extract_names_natasha(text)
spacy_names = extract_names_spacy(text)
print(natasha_names)
print(spacy_names)

# Преобразование результатов в списки строк ФИО
natasha_extracted = natasha_names
spacy_extracted = spacy_names

# Преобразование списков в множества для удобства операций над множествами
natasha_set = set(natasha_names)
spacy_set = set(spacy_names)
golden_set = set(golden_standard)

# Рассчитываем TP (True Positives), FP (False Positives) и FN (False Negatives)
natasha_tp = len(natasha_set.intersection(golden_set))
spacy_tp = len(spacy_set.intersection(golden_set))

natasha_fp = len(natasha_set.difference(golden_set))
spacy_fp = len(spacy_set.difference(golden_set))

natasha_fn = len(golden_set.difference(natasha_set))
spacy_fn = len(golden_set.difference(spacy_set))

# Рассчитываем Precision, Recall и F1-score
natasha_precision = natasha_tp / (natasha_tp + natasha_fp) if (natasha_tp + natasha_fp) > 0 else 0
natasha_recall = natasha_tp / (natasha_tp + natasha_fn) if (natasha_tp + natasha_fn) > 0 else 0
natasha_f1 = 2 * (natasha_precision * natasha_recall) / (natasha_precision + natasha_recall) if (natasha_precision + natasha_recall) > 0 else 0

spacy_precision = spacy_tp / (spacy_tp + spacy_fp) if (spacy_tp + spacy_fp) > 0 else 0
spacy_recall = spacy_tp / (spacy_tp + spacy_fn) if (spacy_tp + spacy_fn) > 0 else 0
spacy_f1 = 2 * (spacy_precision * spacy_recall) / (spacy_precision + spacy_recall) if (spacy_precision + spacy_recall) > 0 else 0

print(f"Natasha Precision: {natasha_precision:.2f}")
print(f"Natasha Recall: {natasha_recall:.2f}")
print(f"Natasha F1-score: {natasha_f1:.2f}")

print(f"SpaCy Precision: {spacy_precision:.2f}")
print(f"SpaCy Recall: {spacy_recall:.2f}")
print(f"SpaCy F1-score: {spacy_f1:.2f}")