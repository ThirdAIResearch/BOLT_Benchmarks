import argparse
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, TFDistilBertForSequenceClassification

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses

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


def train_model(model, dataset, learning_rate, batch_size, epochs):
    train_dataset = dataset['train']
    train_input = train_dataset.to_tf_dataset(columns=["input_ids"], label_cols=["labels"], batch_size=128, shuffle=True)
    optimizer = optimizers.Adam(learning_rate=5e-5)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_input, batch_size=256, epochs=1)
    return model


def evaluate_model(model, dataset, batch_size):
    test_dataset = dataset['test']
    eval_data = test_dataset.to_tf_dataset(columns=["input_ids"], label_cols=["labels"], batch_size=256, shuffle=True)
    model.evaluate(eval_data, batch_size=256)


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = dataset.map(partial(encode_batch, tokenizer=tokenizer, text_key=args.text_key), batched=True)
    config = AutoConfig.from_pretrained(
                MODEL_NAME,
                num_labels=args.num_labels,
            )

    model = TFDistilBertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                config=config,
            )

    train_model(model, dataset, args.learning_rate, args.batch_size, args.num_epochs)
    evaluate_model(model, dataset)


if __name__=='__main__':
    main()



