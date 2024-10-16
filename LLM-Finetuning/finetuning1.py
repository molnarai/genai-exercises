import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

HUGGINGFACE_TOKEN = ''

def prepare_train_dataset(tokenizer, max_length=384):
    dataset = load_dataset("natural_questions", split="train[:1000]")  # Using a subset for demonstration
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1


            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

def prepare_test_dataset(tokenizer, max_length=384):
    dataset = load_dataset("squad_v2", split="validation[:1000]")  # Using a subset for demonstration
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

        inputs["example_id"] = example_ids
        return inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

def compute_metrics(pred):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    predictions = pred.predictions
    labels = pred.label_ids
    
    exact_match = 0
    f1_scores = []
    rouge_l_scores = []
    
    for pred, label in zip(predictions, labels):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        
        # Exact Match
        if pred_text == label_text:
            exact_match += 1
        
        # F1 Score
        pred_tokens = set(pred_text.split())
        label_tokens = set(label_text.split())
        common_tokens = pred_tokens.intersection(label_tokens)
        
        if len(pred_tokens) == 0 or len(label_tokens) == 0:
            f1 = 0
            recall = len(common_tokens) / len(label_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        
        # ROUGE-L Score
        rouge_scores = scorer.score(label_text, pred_text)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
    
    return {
        "exact_match": exact_match / len(predictions),
        "f1_score": np.mean(f1_scores),
        "rouge_l": np.mean(rouge_l_scores),
    }

def main():
    model_name = "decaplusplus/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    train_dataset = prepare_train_dataset(tokenizer)
    test_dataset = prepare_test_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_llama2_qa")
    tokenizer.save_pretrained("./fine_tuned_llama2_qa")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    HUGGINGFACE_TOKEN = open('.env', 'r', encoding='utf-8').read().split('=')[1].replace('"','').strip()
    main()
