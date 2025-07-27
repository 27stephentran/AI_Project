import os
import pandas as pd

def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        label = 1 if label_type == "pos" else 0
        dir_path = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})