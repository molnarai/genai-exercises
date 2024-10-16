import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, pipeline
from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary sentiment analysis

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
train_dataset, val_dataset = tokenized_dataset["train"], tokenized_dataset["test"].train_test_split(test_size=0.1)

# Hyperparameters and Training Configuration
learning_rate = 2e-5
batch_size = 16
num_epochs = 3

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning function using PEFT
def fine_tune(model, train_dataset, optimizer, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            labels = torch.tensor(batch["label"]).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_dataset)}")

# Fine-tune the model using PEFT
fine_tune(model, train_dataset, optimizer, num_epochs, batch_size)

# Save the fine-tuned model
output_dir = "fine_tuned_model/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load the fine-tuned model using the TextClassificationPipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("text-classification", model=output_dir, tokenizer=output_dir, device=device)

# Test the fine-tuned model on a sample text
sample_text = "This movie was fantastic! I loved every bit of it."
result = classifier(sample_text)
print(result)

