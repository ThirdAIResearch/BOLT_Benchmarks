import time
from thirdai import bolt, deployment
import json
import re

class ColdStartCallback(bolt.callbacks.Callback):
    def __init__(self, save_loc, UDT):
        super().__init__()
        self.save_loc = save_loc
        self.model = UDT

    def on_epoch_end(self, model, train_state):
        self.model.save(self.save_loc)
        cur_acc = train_state.get_train_metric_values("categorical_accuracy")[-1]
        if cur_acc > 0.99:
            train_state.stop_training = True
        self.final_cold_start_accuracy = cur_acc


def measure_inference_time(model, tst_file):
    with open(tst_file, "r") as f:
        lines = f.readlines()[1:]

        queries = []
        for line in lines:
            labels, query = line.strip().split(",")
            queries.append(query)
    
    total_time = 0
    for query in queries:
        start = time.time()
        model.predict({"QUERY": query})
        end = time.time()
        total_time += end - start
    
    print("Average query time: ", total_time / len(queries))


def main():
    for dataset, n_classes in [("cooking", 26109), ("serverfault", 316485), ("apple-helpdesk", 123485)]:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "LABEL_IDS": bolt.types.categorical(delimiter=";"),
                "QUERY": bolt.types.text(contextual_encoding="local"),
            },
            n_target_classes=n_classes,
            integer_target=True,
            target="LABEL_IDS",
            model_config=f"{dataset}_model_config",
        )

        callback = ColdStartCallback("lol", model)
        model.cold_start(
            filename=f"{dataset}/reformatted_trn_unsupervised.csv",
            strong_column_names=["TITLE"] if dataset != "cooking" else [],
            weak_column_names=["DESCRIPTION", "BRAND"],
            learning_rate=0.001,
            epochs=15,
            metrics=["precision@1", "categorical_accuracy", "recall@100"],
            callbacks=[callback]
        )

        model.save(f"{dataset}/after_coldstart.bolt")
        model = bolt.UniversalDeepTransformer.load(f"{dataset}/after_coldstart.bolt")

        model.train(
            filename=f"{dataset}/reformatted_trn_supervised.csv",
            learning_rate=0.001,
            epochs=1,
            metrics=["f_measure(0.95)", "categorical_accuracy", "recall@100"]
        )

        model.evaluate(
            filename=f"{dataset}/reformatted_tst_supervised.csv",
            metrics=["precision@1", "categorical_accuracy", "recall@100"]
        )

        measure_inference_time(model, f"{dataset}/reformatted_tst_supervised.csv")


if __name__ == "__main__":
    main()