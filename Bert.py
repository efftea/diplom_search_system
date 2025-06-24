import os
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import streamlit as st
from stqdm import stqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RuBERTWrapper:
    def __init__(self, model_path='rubert_finetuned_search'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.doc_embeddings = None
        self.documents_info = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_long_text(self, text):
        """Кодирование длинных текстов с разбиением на части"""
        # Разбиваем текст на предложения или части
        parts = self._split_text(text)
        part_embeddings = []

        for part in parts:
            inputs = self.tokenizer(part, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                part_embeddings.append(embedding.cpu().numpy())

        # Усредняем эмбеддинги всех частей
        return np.mean(part_embeddings, axis=0)

    def _split_text(self, text, max_length=500):
        """Разбивает текст на части по предложениям или по длине"""
        sentences = text.split('.')
        parts = []
        current_part = ""

        for sentence in sentences:
            if len(current_part) + len(sentence) < max_length:
                current_part += sentence + "."
            else:
                if current_part:
                    parts.append(current_part)
                current_part = sentence + "."

        if current_part:
            parts.append(current_part)

        return parts

    def index_documents(self, documents):
        """Индексация документов с обработкой длинных текстов"""
        texts = [doc[1] for doc in documents]
        embeddings = []

        for text in stqdm(texts, desc="Индексация документов"):
            if len(self.tokenizer.tokenize(text)) > 500:  # Примерный порог для длинных документов
                embedding = self._encode_long_text(text)
            else:
                inputs = self.tokenizer(text, padding=True, truncation=True,
                                        max_length=512, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
            embeddings.append(embedding)

        self.doc_embeddings = np.vstack(embeddings)
        self.documents_info = documents

    def search(self, query, top_n=10):
        if self.doc_embeddings is None:
            raise ValueError("Документы не индексированы!")

        # Кодирование запроса (может быть длинным)
        if len(self.tokenizer.tokenize(query)) > 500:
            query_embedding = self._encode_long_text(query)
        else:
            inputs = self.tokenizer(query, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                query_embedding = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()

        # Нормализация векторов
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings_norm = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)

        # Расчет косинусной схожести
        similarities = np.dot(query_embedding, doc_embeddings_norm.T)[0]

        # Выбор топ-N результатов
        top_indices = np.argsort(similarities)[::-1][:top_n]

        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]


def load_documents_from_folder(folder_path: str) -> list:
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
    st.title("Поисковик на RuBERT для русскоязычных текстов")

    folder_path = st.text_input("Путь к папке с документами", value="Documents")

    if st.button("Загрузить и проиндексировать документы"):
        documents = load_documents_from_folder(folder_path)
        if not documents:
            st.warning("Не найдено документов в указанной папке.")
            return

        st.info(f"Найдено документов: {len(documents)}. Начинаю индексацию...")

        rubert = RuBERTWrapper()

        with st.spinner('Индексация документов...'):
            rubert.index_documents(documents)

        st.success("Индексация завершена!")
        st.session_state['searcher'] = rubert
        st.session_state['documents'] = documents

    if 'searcher' in st.session_state and 'documents' in st.session_state:
        query = st.text_input("Введите поисковый запрос")
        if st.button("Поиск") and query.strip():
            results = st.session_state['searcher'].search(query)
            if not results:
                st.warning("Ничего не найдено.")
            else:
                st.write(f"Результаты поиска для запроса: **{query}**")
                for rank, (doc_index, score) in enumerate(results, 1):
                    filename = st.session_state['documents'][doc_index][0]
                    content = st.session_state['documents'][doc_index][1]

                    st.write(f"{rank}. **{filename}** (релевантность: {score:.4f})")

                    # Показываем полный документ с возможностью развернуть/свернуть
                    with st.expander("Показать полный документ"):
                        st.text(content[:10000])  # Ограничиваем вывод для производительности


if __name__ == "__main__":
    run_search_app()