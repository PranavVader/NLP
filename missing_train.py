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

def train_model(dataset, epochs=2, batch_size=8, learning_rate=2e-5):
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

    model.save_pretrained('fine_tuned_model5')
    tokenizer.save_pretrained('fine_tuned_model5')
    print("Model saved to 'fine_tuned_model5' folder")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    synthetic_df = pd.read_csv("C:/Users/asust/OneDrive/Documents/synthetic_missing_info_dataset3.csv")
    # external_df = pd.read_csv("C:/Users/asust/Downloads/archive/winograd_train.csv")

    train_df, test_df = train_test_split(synthetic_df, test_size=0.2)

    train_dataset = load_dataset_from_df(train_df, tokenizer, text_col='sentence', label_col='missing_info_label')
    test_dataset = load_dataset_from_df(test_df, tokenizer, text_col='sentence', label_col='missing_info_label')

    train_model(train_dataset)

model_path = "fine_tuned_model5"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_df['missing_info_label'] = test_df['missing_info_label'].astype(int) 

correct = 0
total = 0

for idx, row in test_df.iterrows():
    inputs = tokenizer(row['sentence'], return_tensors='pt', truncation=True, max_length=256, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = outputs.logits.argmax(dim=1).item()

    if predicted_label == row['missing_info_label']:
        correct += 1
    total += 1

accuracy = correct / total if total else 0
print(f"Test Accuracy: {accuracy:.4f}")
