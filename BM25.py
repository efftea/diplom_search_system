import os
import math
from collections import defaultdict
from typing import List, Tuple
import pymorphy3
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import streamlit as st
from stqdm import stqdm  # tqdm для Streamlit

# Класс BM25 для русского языка
class BM25Russian:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_freqs = defaultdict(int)
        self.word_doc_count = defaultdict(int)
        self.idf = {}
        self.doc_len = []
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        self.tokenizer = RegexpTokenizer(r'\w+')

    def preprocess(self, text: str) -> List[str]:
        try:
            text = text.lower()
            tokens = self.tokenizer.tokenize(text)
            processed_tokens = []
            for token in tokens:
                if token.isalpha() and token not in self.stop_words:
                    try:
                        lemma = self.morph.parse(token)[0].normal_form
                        processed_tokens.append(lemma)
                    except Exception:
                        continue
            return processed_tokens
        except Exception as e:
            st.error(f"Ошибка при обработке текста: {e}")
            return []

    def add_document(self, document: str):
        tokens = self.preprocess(document)
        self.documents.append(tokens)
        self.corpus_size += 1
        self.doc_len.append(len(tokens))

        frequencies = defaultdict(int)
        for word in tokens:
            frequencies[word] += 1

        for word, freq in frequencies.items():
            self.word_doc_count[word] += 1
            self.doc_freqs[(self.corpus_size - 1, word)] = freq

        self.avg_doc_len = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

    def initialize(self):
        for word in self.word_doc_count:
            self.idf[word] = math.log(
                (self.corpus_size - self.word_doc_count[word] + 0.5) /
                (self.word_doc_count[word] + 0.5) + 1
            )

    def get_score(self, query: List[str], doc_index: int) -> float:
        score = 0.0
        doc_len = self.doc_len[doc_index]

        for word in query:
            if word not in self.word_doc_count:
                continue

            freq = self.doc_freqs.get((doc_index, word), 0)
            numerator = self.idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += numerator / denominator

        return score

    def search(self, query: str, top_n=10) -> List[Tuple[int, float]]:
        query_terms = self.preprocess(query)
        scores = []

        for doc_index in range(self.corpus_size):
            score = self.get_score(query_terms, doc_index)
            scores.append((doc_index, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def load_documents_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"Папка '{folder_path}' не найдена.")
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append((filename, content))
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='cp1251') as file:
                        content = file.read()
                        documents.append((filename, content))
                except Exception as e:
                    st.warning(f"Не удалось прочитать файл {filename}: {e}")
            except Exception as e:
                st.warning(f"Ошибка при чтении файла {filename}: {e}")
    return documents


def run_search_app():
    st.title("Поисковик на BM25 для русскоязычных текстов")

    folder_path = st.text_input("Путь к папке с документами", value="Documents")

    if st.button("Загрузить и проиндексировать документы"):
        documents = load_documents_from_folder(folder_path)
        if not documents:
            st.warning("Не найдено документов в указанной папке.")
            return

        st.info(f"Найдено документов: {len(documents)}. Начинаю индексацию...")
        bm25 = BM25Russian()

        for i, (filename, content) in enumerate(stqdm(documents, desc="Индексация документов"), 1):
            bm25.add_document(content)

        bm25.initialize()
        st.success("Индексация завершена!")

        st.session_state['bm25'] = bm25
        st.session_state['documents'] = documents

    if 'bm25' in st.session_state and 'documents' in st.session_state:
        query = st.text_input("Введите поисковый запрос")
        if st.button("Поиск") and query.strip():
            results = st.session_state['bm25'].search(query)
            if not results or all(score == 0 for _, score in results):
                st.warning("Ничего не найдено.")
            else:
                st.write(f"Результаты поиска для запроса: **{query}**")
                for rank, (doc_index, score) in enumerate(results, 1):
                    if score > 0:
                        filename = st.session_state['documents'][doc_index][0]
                        snippet = st.session_state['documents'][doc_index][1][:300].replace('\n', ' ')
                        st.write(f"{rank}. **{filename}** (релевантность: {score:.4f})")
                        st.write(f"> {snippet}...")

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    run_search_app()