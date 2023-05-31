import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Load the data
dataframe = pd.read_csv('../csv_files/processed_data/preprocessed_data.tsv', sep='\t')
dataframe['cleaned_title'] = dataframe['cleaned_title'].astype(str)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokens = [tokenizer.encode(text, add_special_tokens=True) for text in dataframe['cleaned_title']]
max_length = max([len(token) for token in tokens])
input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True, padding='max_length',
                                            max_length=max_length, truncation=True) for text in dataframe['cleaned_title']])

# Build attention masks
attention_masks = []
for sequence in input_ids:
    mask = [float(value > 0) for value in sequence]
    attention_masks.append(mask)
attention_masks = torch.tensor(attention_masks)

# Prepare data loaders
batch_size = 32
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(dataframe['cluster'].values))
train_samples = int(0.8 * len(dataset))
val_samples = len(dataset) - train_samples
train_data, val_data = random_split(dataset, [train_samples, val_samples])
train_data_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_data_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=False,
                                                      output_hidden_states=False)

# Tune the model
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_epochs = 4
total_steps = len(train_data_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

hardware = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(hardware)

for epoch in range(num_epochs):
    for batch in train_data_loader:
        batch_input_ids = batch[0].to(hardware)
        batch_input_mask = batch[1].to(hardware)
        batch_labels = batch[2].to(hardware)
        model.zero_grad()
        output = model(batch_input_ids, attention_mask=batch_input_mask, labels=batch_labels)
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

# Load test data
test_dataframe = pd.read_csv('../csv_files/data/post_from_reddit_main.tsv', sep='\t')
test_dataframe['cleaned_title'] = test_dataframe['cleaned_title'].astype(str)
test_inputs = torch.tensor([tokenizer.encode(text, add_special_tokens=True, padding='max_length',
                                              max_length=max_length, truncation=True) for text in test_dataframe['cleaned_title']])
test_masks = []
for seq in test_inputs:
    seq_mask = [float(i > 0) for i in seq]
    test_masks.append(seq_mask)
test_masks = torch.tensor(test_masks)
test_labels = torch.tensor(test_dataframe['cluster'].values)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_data_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Evaluate model on test set
predictions, actual_labels = [], []
for batch in test_data_loader:
    batch_input_ids = batch[0].to(hardware)
    batch_input_mask = batch[1].to(hardware)
    batch_labels = batch[2].to(hardware)
    with torch.no_grad():
        output = model(batch_input_ids, attention_mask=batch_input_mask)
    logits = output[0]
    logits = logits.detach().cpu().numpy()
    label_ids = batch_labels.to('cpu').numpy()
    predictions.extend(logits)
    actual_labels.extend(label_ids)
predicted_labels = np.argmax(np.concatenate(predictions, axis=0), axis=1)
actual_classes = np.concatenate(actual_labels, axis=0)
accuracy_score = np.mean(predicted_labels == actual_classes)
print(f'Test Accuracy: {accuracy_score:.3f}')
