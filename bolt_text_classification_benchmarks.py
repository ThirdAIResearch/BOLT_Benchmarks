from thirdai import bolt
import sys
import pandas as pd
import time


def train_bolt(train_data: str, test_data: str, n_targets: int, learning_rates=[0.001, 0.001, 0.0001]):
    model = bolt.UniversalDeepTransformer(
      data_types={
        "text": bolt.types.text(),
        "category": bolt.types.categorical(),
      },
      target="category",
      n_target_classes=n_targets,
      integer_target=True,
      delimiter=','
    )

    for learning_rate in learning_rates:
        model.train(train_data, learning_rate=learning_rate, epochs=1)
        model.evaluate(test_data, metrics=["categorical_accuracy"])

    df = pd.read_csv(test_data)
    test_samples = df["text"].iloc[:1000]

    start = time.perf_counter()
    for text in test_samples:
        model.predict({"text": text})
    end = time.perf_counter()

    print(f"Inference time: {1000 * (end - start) / len(test_samples)} ms")


if __name__ == "__main__":
    train_bolt(
      train_data="/share/data/amazon_polarity/train.csv",
      test_data="/share/data/amazon_polarity/test.csv",
      n_targets=2,
    )

    train_bolt(
      train_data="/share/data/yelp_polarity/train.csv",
      test_data="/share/data/yelp_polarity/test.csv",
      n_targets=2,
    )

    train_bolt(
      train_data="/share/data/dbpedia/train.csv",
      test_data="/share/data/dbpedia/test.csv",
      n_targets=14,
    )

    train_bolt(
      train_data="/share/data/ag_news/train.csv",
      test_data="/share/data/ag_news/test.csv",
      n_targets=4,
    )

    train_bolt(
      train_data="/share/data/twitter_eval/train.csv",
      test_data="/share/data/twitter_eval/test.csv",
      n_targets=20,
      learning_rates=[0.01, 0.001, 0.001]
    )