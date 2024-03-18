import csv
import jsonlines

# Открываем CSV-файл для чтения
with open('datasets/technology_ner/technologies_sources/user-languages.csv', newline='') as csvfile:
    # Создаем объект чтения CSV
    reader = csv.reader(csvfile)

    # Получаем первую строку, которая содержит заголовки столбцов
    headers = next(reader)

technologies_list = headers

# Формирование структуры для JSONL
data = []
for technology in technologies_list:
    words = technology.split()  # Разделение имени на отдельные слова
    labels = (["TECHNOLOGY"] * (len(words)))  # Формирование меток

    entry = {"words": words, "labels": labels}
    data.append(entry)

# Запись в JSONL файл
with jsonlines.open('datasets/technology_ner/technologies.jsonl', mode='w') as writer:
    for entry in data:
        writer.write(entry)