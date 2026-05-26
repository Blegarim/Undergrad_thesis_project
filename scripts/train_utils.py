"""
Shared training utilities used by train.py and train_two_phase.py.
"""

import os
import time
import lmdb
import psutil
import torch


MAX_SEQ_LEN = 20  # cap temporal length to bound VRAM across chunks


def collate_fn(batch):
    images_tight = torch.stack([item['images_tight'][:MAX_SEQ_LEN] for item in batch], dim=0)
    images_context = torch.stack([item['images_context'][:MAX_SEQ_LEN] for item in batch], dim=0)
    motions = torch.stack([item['motions'][:MAX_SEQ_LEN] for item in batch], dim=0)[..., :8]
    labels = {k: torch.stack([item[k] for item in batch], dim=0) for k in ['actions', 'looks', 'crosses']}
    return images_tight, images_context, motions, labels


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def remap_cross_labels(labels: dict) -> None:
    """Clamp crosses labels to binary [0, 1] in-place."""
    labels['crosses'] = torch.clamp(labels['crosses'], 0, 1)


def gather_chunks(folders):
    """
    Collect *.lmdb files from one or more folders.
    Missing folders are skipped with a warning. Raises only if no folder exists,
    so callers can pass optional variants (e.g. augmented data) without crashing
    on a fresh machine.
    """
    if isinstance(folders, str):
        folders = [folders]
    all_files = []
    missing = []
    for folder in folders:
        if not os.path.isdir(folder):
            missing.append(folder)
            continue
        chunk_files = sorted([os.path.join(folder, f)
                              for f in os.listdir(folder)
                              if f.endswith('.lmdb')])
        all_files.extend(chunk_files)
    if missing:
        print(f"gather_chunks: skipping missing folder(s): {missing}")
    if not all_files:
        raise FileNotFoundError(
            f"gather_chunks: no .lmdb chunks found in any of {folders}"
        )
    return all_files


def wait_for_memory(threshold=96, interval=1):
    while psutil.virtual_memory().percent > threshold:
        print(f"RAM at {psutil.virtual_memory().percent:.1f}%, waiting...")
        time.sleep(interval)


def mp_async_load(idx, path, queue):
    """
    Warm LMDB chunk file (light read) in a background process, then return the path.
    Opens the LMDB and reads one _meta key to encourage OS file caching,
    then passes back the path string for the parent to instantiate LMDBChunkDataset.
    """
    try:
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key.decode().endswith("_meta"):
                    _ = txn.get(key)
                    break
        env.close()
        queue.put((idx, 'ok', path))
    except Exception as e:
        queue.put((idx, 'err', str(e)))
