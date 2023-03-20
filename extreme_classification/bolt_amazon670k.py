import thirdai
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="Amazon670k/train.txt",
        help="The path to the training dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="Amazon670k/test.txt",
        help="The path to the testing dataset.",
    )
    args = parser.parse_args()
    return args.train_path, args.test_path


def train_model(train_file, test_file, num_epochs, learning_rate):
    model = thirdai.bolt.UniversalDeepTransformer(
        file_format="svm", n_target_classes=670091, input_dim=135909
    )
    for _ in range(num_epochs):
        model.train(
            train_file,
            epochs=num_epochs,
            learning_rate=learning_rate,
            metrics=["categorical_accuracy"],
        )
        model.evaluate(test_file, metrics=["categorical_accuracy"])
    return model


def sparse_eval(model, test_file):
    print(
        model.evaluate(
            test_file, use_sparse_inference=True, metrics=["categorical_accuracy"]
        )
    )


if __name__ == "__main__":
    train_file, test_file = parse_args()
    model = train_model(
        train_file=train_file, test_file=test_file, num_epochs=1, learning_rate=0.0001
    )
    sparse_eval(model, test_file)
