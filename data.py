import os
import numpy as np
import tiktoken
from datasets import load_dataset

# We place the function outside the guard so child cores can see it
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids}

# --- THE WINDOWS FIX ---
# This guard prevents the child CPU cores from infinitely spawning more cores
if __name__ == '__main__':
    
    print("Downloading TinyStories dataset (5% subset)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")

    print("Splitting into train and val...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    print("Tokenizing the stories into numbers...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing splits",
        num_proc=4, # Safely uses 4 cores now!
    )

    for split, dset in tokenized.items():
        arr_len = sum(len(x) for x in dset['ids'])
        print(f"Saving {split}.bin with {arr_len:,} tokens...")
        
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in dset:
            ids = example['ids']
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
        
    print("Data preparation complete! You now have train.bin and val.bin ready for the GPU.")
