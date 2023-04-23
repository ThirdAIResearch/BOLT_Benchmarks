import thirdai
import argparse
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["Amazon670k", "Delicious200K", "Wiki350K"],
        help="The dataset to train on.",
    )
    parser.add_argument(
        "--train_path",
        default="",
        help="The path to the training dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="",
        help="The path to the testing dataset.",
    )
    parser.add_argument("--test_subsample_ratio", type=float, default=0.1)
    args = parser.parse_args()

    folder = Path(args.dataset)

    if args.train_path == "":
        args.train_path = folder / "train.txt"
    if args.test_path == "":
        args.test_path = folder / "test.txt"

    sampled_test_path = folder / "sampled_test.txt"
    sampled_test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sampled_test_path, "w") as sampled_test_file:
        with open(args.test_path) as test_file:
            for line in test_file:
                if random.uniform(0, 1) < args.test_subsample_ratio:
                    sampled_test_file.write(line)

    return str(args.train_path), str(sampled_test_path)


def train_model(train_file, test_file, num_epochs, learning_rate):
    model = thirdai.bolt.UniversalDeepTransformer(
        file_format="svm", n_target_classes=670091, input_dim=135910
    )
    for _ in range(num_epochs):
        model.train(
            train_file,
            epochs=num_epochs,
            learning_rate=learning_rate,
            metrics=["categorical_accuracy"],
        )
        print(
            model.evaluate(
                test_file, metrics=["categorical_accuracy"], return_metrics=True
            )
        )
    return model


def sparse_eval(model, test_file):
    print(
        model.evaluate(
            test_file,
            use_sparse_inference=True,
            metrics=["categorical_accuracy"],
            return_metrics=True,
        )
    )


if __name__ == "__main__":
    train_file, test_file = parse_args()
    model = train_model(
        train_file=train_file, test_file=test_file, num_epochs=5, learning_rate=0.0001
    )
    sparse_eval(model, test_file)
