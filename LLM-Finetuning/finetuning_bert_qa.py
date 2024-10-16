import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset



def main(model_name='bert-base-uncased', dataset_name='squad_v2'):

    # Dataset
    dataset = load_dataset(dataset_name)  # Or your custom dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["question"], examples["context"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
   
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == '__main__':
    main()