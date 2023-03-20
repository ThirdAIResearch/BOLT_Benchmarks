import gdown
import shutil
import os

def download_dataset():
    train_file_loc = "Amazon670k/train.txt"
    test_file_loc = "Amazon670k/test.txt"
    if not os.path.exists(train_file_loc) or not os.path.exists(test_file_loc):
        gdown.download(
            "https://drive.google.com/u/0/uc?id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR&export=download",
            output="Amazon670k.zip",
        )
        shutil.unpack_archive("Amazon670k.zip", "Amazon670k")
    return train_file_loc, test_file_loc


if __name__ == "__main__":
    train_file, test_file = download_dataset()
