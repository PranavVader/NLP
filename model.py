import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset , DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

class RequirementDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx],
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_dataset_from_df(df, tokenizer, text_col, label_col):
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    return RequirementDataset(texts, labels, tokenizer)

def train_model(dataset, epochs=3, batch_size=8, learning_rate=2e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs} Start ")
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} END | Average Loss: {avg_loss:.4f}")
        print(f"Completed Epoch {epoch + 1}/{epochs}\n")

    model.save_pretrained('fine_tuned_model')
    tokenizer.save_pretrained('fine_tuned_model')
    print("Model saved to 'fine_tuned_model' folder")

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    synthetic_df = pd.read_csv("C:/Users/asust/OneDrive/Documents/synthetic_missing_info_dataset.csv")
    external_df = pd.read_csv("C:/Users/asust/Downloads/archive/winograd_train.csv")

    synthetic_dataset = load_dataset_from_df(synthetic_df, tokenizer, text_col='sentence', label_col='missing_info_label')
    external_dataset = load_dataset_from_df(external_df, tokenizer, text_col='Text', label_col='A-coref')

    combined_dataset = ConcatDataset([synthetic_dataset, external_dataset])

    train_model(combined_dataset)