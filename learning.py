import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from typing import List, Dict, Union

MODEL_NAME = 'DeepPavlov/rubert-base-cased'
SAVE_PATH = 'rubert_finetuned_search'
TRAIN_DATA_PATH = 'train_data.json'
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
DOC_STRIDE = 128


class SearchDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = MAX_LENGTH,
                 doc_stride: int = DOC_STRIDE):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

    def __len__(self):
        return len(self.data)

    def _process_long_document(self, query: str, document: str) -> List[Dict]:
        tokenized = self.tokenizer(
            query,
            document,
            truncation='only_second',
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False
        )

        samples = []
        for i in range(len(tokenized['input_ids'])):
            samples.append({
                'input_ids': tokenized['input_ids'][i],
                'attention_mask': tokenized['attention_mask'][i],
                'token_type_ids': tokenized['token_type_ids'][i]
            })
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        try:
            encoding = self.tokenizer(
                item['query'],
                item['document'],
                padding='max_length',
                truncation='only_second' if len(item['document']) > 1000 else True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        except:
            chunks = self._process_long_document(item['query'], item['document'])
            encoding = {k: torch.tensor([chunks[0][k]]) for k in chunks[0]}
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def load_train_data() -> List[Dict]:
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def fine_tune_rubert():
    train_data = load_train_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    dataset = SearchDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                'labels': batch['labels'].to(device)
            }

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Модель сохранена в {SAVE_PATH}")


if __name__ == "__main__":
    fine_tune_rubert()