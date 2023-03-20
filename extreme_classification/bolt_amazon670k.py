
import thirdai
from extreme_download import download_amazon_670k


def train_model(train_file, test_file, num_epochs, learning_rate):
    model = thirdai.bolt.UniversalDeepTransformer(file_format="svm", n_target_classes=670091, input_dim=135909)
    model.train(train_file, epochs=num_epochs, learning_rate=learning_rate)
    metrics = model.evaluate(test_file, metrics=["categorical_accuracy"])
    print(metrics)

if __name__ == "__main__":
    train_file, test_file = download_amazon_670k()
    train_model(train_file=train_file, test_file=test_file, num_epochs=3, learning_rate=0.0001)
