import argparse
from functools import partial
import mlflow
import os
import platform
import socket
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments,\
        Trainer, EvalPrediction

import numpy as np

MODEL_NAME = "distilbert-base-uncased"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="The string identifier of the dataset as listed on Huggingface")
    parser.add_argument("--text_key", default="text", help="The key name associated with the text data")
    parser.add_argument("--num_labels", default=2, type=int, help="The number of output classes in the given task")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of passes through the data for fine-tuning")
    args = parser.parse_args()
    return args


def encode_batch(batch, tokenizer, text_key):
    """Encodes a batch of input data using the model tokenizer."""
    tokens = tokenizer(batch[text_key], max_length=80, truncation=True, padding="max_length")
    tokens['labels'] = batch['label']  # The HuggingFace model expects the target to be called 'labels'
    return tokens


def compute_accuracy(preds):
    top_predictions = np.argmax(preds.predictions, axis=1)
    return {"acc": (top_predictions == preds.label_ids).mean()}


def train_model(model, dataset, learning_rate, num_epochs, batch_size):
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10000000,
        save_strategy="no",
        output_dir="./training_output",
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    return trainer


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = dataset.map(partial(encode_batch, tokenizer=tokenizer, text_key=args.text_key), batched=True)
    config = AutoConfig.from_pretrained(
                MODEL_NAME,
                num_labels=args.num_labels,
            )

    model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                config=config,
            )
    
    trainer = train_model(model, dataset, args.learning_rate, args.num_epochs, args.batch_size)
    output = trainer.evaluate()


if __name__=='__main__':
    main()



