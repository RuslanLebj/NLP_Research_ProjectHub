from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from yake import KeywordExtractor
import docx2txt
from package_NLP import text_preprocessing, line_splitter, lines_to_sentences_tokenization, \
    introduction_conclusion_extract
from sklearn.metrics import f1_score

# Текстовые данные для извлечения именованных сущностей ФИО
with open('test_set_keywords.txt', 'r') as file:
    text_for_keywords = file.read()

# Количество извлекаемых слов каждой моделью:
top_n = 10

# Экстрактору TF-IDF модели зададим параметры:
# Фраза размерносью 2
tfidfModel = TfidfVectorizer(ngram_range=(1, 1))
# Экстрактору Rake модели зададим параметры:
# По умолчанию, так как она не имеет поддержки установки максимального размера фразы
rakeModel = Rake()
# Экстрактору Yake модели зададим параметры:
# Русскоязычная модель, фраза размерностью 2, мера схожести слов не более 50%, извелчение в количестве 10 фраз
yakeModel = KeywordExtractor(lan="ru", n=1, dedupLim=0.5, top=top_n)

# Извечем ключевые слова:
tfidfModel.fit([text_for_keywords])
tfidf_feature_names = tfidfModel.get_feature_names_out()
# Получаем веса TF-IDF для каждого слова
tfidf_weights = tfidfModel.transform([text_for_keywords]).toarray()[0]
# Сортируем слова по весам TF-IDF и выводим ключевые слова
tfidf_keywords = [(tfidf_feature_names[idx], tfidf_weights[idx]) for idx in tfidf_weights.argsort()[::-1]]

rakeModel.extract_keywords_from_text(text_for_keywords)
rake_keywords = rakeModel.get_ranked_phrases_with_scores()

yake_keywords = yakeModel.extract_keywords(text_for_keywords)

# Результаты:
print("TF-IDF:")
for weight, keyword in tfidf_keywords[:top_n]:
    print(f"{keyword} - {weight}")

print("Rake:")
for score, keyword in rake_keywords[:top_n]:
    print(f"{keyword} - {score}")

print("Yake:")
for score, keyword in yake_keywords:
    print(f"{keyword} - {score}")

# Золотой стандарт (список выбранных экспертом ключевых слов)
gold_standard_keywords = ["социальной сети", "соцсети", "modeus", "университета", "элективы", "Вконтакте", 'рамках', 'элективных', 'курсов', 'Modeus', 'сети''управления', 'Modeus', 'социальной', 'элективных', 'сети', 'рамках', 'Выбор', 'управления', 'курсов', 'университета']
gold_standard_keywords = set(gold_standard_keywords)

# Преобразуйте извлеченные ключевые слова в списки слов для сравнения и удаляем дубликаты
tfidf_extracted_keywords = set([keyword for keyword, _ in tfidf_keywords[:top_n]])
rake_extracted_keywords = [_.split() for keyword, _ in rake_keywords[:top_n]]
rake_extracted_keywords = set(sum(rake_extracted_keywords, []))
yake_extracted_keywords = set([keyword for keyword, _ in yake_keywords])

# TFIDF
# Рассчитываем TP (True Positives), FP (False Positives) и FN (False Negatives)
tfidf_tp = len(tfidf_extracted_keywords.intersection(gold_standard_keywords))
tfidf_fp = len(tfidf_extracted_keywords.difference(gold_standard_keywords))
tfidf_fn = len(gold_standard_keywords.difference(tfidf_extracted_keywords))

# Рассчитываем Precision, Recall и F1-score
tfidf_precision = tfidf_tp / (tfidf_tp + tfidf_fp) if (tfidf_tp + tfidf_fp) > 0 else 0
tfidf_recall = tfidf_tp / (tfidf_tp + tfidf_fn) if (tfidf_tp + tfidf_fn) > 0 else 0
tfidf_f1 = 2 * (tfidf_precision * tfidf_recall) / (tfidf_precision + tfidf_recall) if (tfidf_precision + tfidf_recall) > 0 else 0


# RAKE
# Рассчитываем TP (True Positives), FP (False Positives) и FN (False Negatives)
rake_tp = len(rake_extracted_keywords.intersection(gold_standard_keywords))
rake_fp = len(rake_extracted_keywords.difference(gold_standard_keywords))
rake_fn = len(gold_standard_keywords.difference(rake_extracted_keywords))

# Рассчитываем Precision, Recall и F1-score
rake_precision = rake_tp / (rake_tp + rake_fp) if (rake_tp + rake_fp) > 0 else 0
rake_recall = rake_tp / (rake_tp + rake_fn) if (rake_tp + rake_fn) > 0 else 0
rake_f1 = 2 * (rake_precision * rake_recall) / (rake_precision + rake_recall) if (rake_precision + rake_recall) > 0 else 0


# YAKE
# Рассчитываем TP (True Positives), FP (False Positives) и FN (False Negatives)
yake_tp = len(yake_extracted_keywords.intersection(gold_standard_keywords))
yake_fp = len(yake_extracted_keywords.difference(gold_standard_keywords))
yake_fn = len(gold_standard_keywords.difference(yake_extracted_keywords))

# Рассчитываем Precision, Recall и F1-score
yake_precision = yake_tp / (yake_tp + yake_fp) if (yake_tp + yake_fp) > 0 else 0
yake_recall = yake_tp / (yake_tp + yake_fn) if (yake_tp + yake_fn) > 0 else 0
yake_f1 = 2 * (yake_precision * yake_recall) / (yake_precision + yake_recall) if (yake_precision + yake_recall) > 0 else 0


print(f"metric: TF-IDF  F1-score: {tfidf_f1:.2f}  Precision: {tfidf_precision:.2f}  Recall: {tfidf_recall:.2f}")
print(f"metric: RAKE  F1-score: {rake_f1:.2f}  Precision: {rake_precision:.2f}  Recall: {rake_recall:.2f}")
print(f"metric: YAKE  F1-score: {yake_f1:.2f}  Precision: {yake_precision:.2f}  Recall: {yake_recall:.2f}")
