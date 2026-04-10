"""
Prepare the Tiny Shakespeare dataset for character-level language modeling.

It will create (in the same folder):
  data/input.txt  
  data/train.bin        -> 
  data/val.bin          -> binary files containing the tokenized data for training and validation
  data/meta.pkl         -> contains char to index mapping for use in prompt.py and get vocab size for train.py
"""

import os
import pickle
import numpy as np
from urllib.request import urlopen


input = "Shakespeare"
#input = "Gameofthrones"



DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(DATA_DIR, "input.txt")

if input == "Gameofthrones":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # points to project_root/data
    INPUT_FILE = os.path.join(DATA_DIR, "gameofthrones.txt")


#INPUT_FILE = os.path.join(DATA_DIR, "gameofthrones.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

VAL_FRACTION = 0.9  # 90% train / 10% val
TRAIN_FRACTION = 1.0

def download_if_missing():
    if input == "Gameofthrones":
        return
    if os.path.exists(INPUT_FILE):
        return
    print("input.txt not found. Downloading Tiny Shakespeare...")
    with urlopen(DATA_URL) as r:
        text = r.read().decode("utf-8")
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Downloaded to {INPUT_FILE}")

def create_dataset(data, val_fraction, train_fraction):
    n = len(data)
    n_val = int(n * val_fraction)

    train_data = data[:int(n_val*train_fraction)]
    val_data = data[n_val:n]
    return train_data, val_data


def main(train_fraction=TRAIN_FRACTION):
    download_if_missing()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = f.read()

    chars = sorted(set(data))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    train_data, val_data = create_dataset(
        data,
        val_fraction=VAL_FRACTION,
        train_fraction=train_fraction
    )

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    suffix = str(int(train_fraction * 100))  # 10, 25, 50, ...

    train_path = os.path.join(DATA_DIR, f"shakespeare_{suffix}_train.bin")
    val_path = os.path.join(DATA_DIR, f"shakespeare_{suffix}_val.bin")
    meta_path = os.path.join(DATA_DIR, f"shakespeare_{suffix}_meta.pkl")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "train_fraction": train_fraction,
        "dataset": "tiny_shakespeare_char",
        "source_url": DATA_URL,
    }

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"train has {len(train_ids):,} tokens") 
    print(f"val has {len(val_ids):,} tokens")
    print(f"Done for {train_fraction}")
    print(f"Wrote: {train_path}, {val_path}, {meta_path}")


#if __name__ == "__main__":
#    main()


#%%

fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

for frac in fractions:
    print(f"\n Running fraction: {frac} \n")
    
    main(train_fraction=frac)