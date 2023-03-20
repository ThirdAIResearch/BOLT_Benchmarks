import gdown
import shutil
import os

def download_amazon_670k():
    train_file_loc = "Amazon670k/train.txt"
    test_file_loc = "Amazon670k/test.txt"
    if not os.path.exists(train_file_loc) or not os.path.exists(test_file_loc):
        gdown.download(
            "https://drive.google.com/u/0/uc?id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR&export=download",
            output="Amazon670k.zip",
        )
        shutil.unpack_archive("Amazon670k.zip")
        shutil.move("Amazon670k.bow", "Amazon670k")
        os.remove("Amazon670k.zip")
    return train_file_loc, test_file_loc


def download_amazon_670k():
    train_file_loc = "Amazon670k/train.txt"
    test_file_loc = "Amazon670k/test.txt"
    if not os.path.exists(train_file_loc) or not os.path.exists(test_file_loc):
        gdown.download(
            "https://drive.google.com/u/0/uc?id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR&export=download",
            output="Amazon670k.zip",
        )
        shutil.unpack_archive("Amazon670k.zip")
        shutil.move("Amazon670k.bow", "Amazon670k")
        os.remove("Amazon670k.zip")
    return train_file_loc, test_file_loc