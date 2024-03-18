import nltk
from string import punctuation
import docx2txt
import pandas as pd
import re
import os
from package_NLP import text_preprocessing, line_splitter, chapter_selector, lines_to_sentences_tokenization, \
    df_sentences_preprocessing


path_in = "projects/test"
path_out = "classifier_sets/test_set"
if not os.path.exists(f"{path_out}"):
    os.makedirs(f"{path_out}")
for filename in os.listdir(f'{path_in}'):
    text = docx2txt.process(f'{path_in}/{filename}')
    text_lines = line_splitter(text_preprocessing(text))
    chapter_number = 4
    chapter_lines = chapter_selector(text_lines, chapter_number)
    chapter_sentences = lines_to_sentences_tokenization(chapter_lines)
    df_preprocessed_sentences = df_sentences_preprocessing(
        pd.DataFrame({"sentence": chapter_sentences, "class": [0 for _ in range(len(chapter_sentences))]}))
    df_preprocessed_sentences.to_csv(f'{path_out}/{filename[:-5]}.csv', encoding="windows-1251", index=False, sep=';')

