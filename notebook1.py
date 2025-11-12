import os
path = r"preprocessed_train_lmdb/chunk_000000.lmdb"
size_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
print(f"Actual disk used: {size_bytes/1024**3:.2f} GB")
