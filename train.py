import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models.Vision_Transformer import ViT_Hierarchical
from models.Regression import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config

import time
import gc
import random
import csv
import psutil
import multiprocessing as mp
import multiprocessing.queues as mpq
from queue import Empty
from tqdm import tqdm
from datetime import datetime

'''
Training script for the PIE dataset using an Ensemble Model with Temporal ConvNet-GRU-Attention, Hierarchical Vision Transformer and Cross Attention.
I love femboys. 
'''

def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :4]
    labels = {k: torch.stack([item[k] for item in batch], dim=0) for k in ['actions', 'looks', 'crosses']}
    return images, motions, labels

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

def remap_cross_labels(labels):
    crosses = labels['crosses']
    crosses = torch.clamp(crosses, 0, 1)
    labels['crosses'] = crosses

def filter_irrelevant(data):
    return [item for item in data if int(item['crosses'].item())==0 or int(item['crosses'].item())==1]

def class_weight(a, b, device):
    y = np.array([0]*a + [1]*b)
    weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return torch.tensor(weight, dtype=torch.float).to(device)

def train_one_chunk(model, dataloader, criterion, optimizer, device, loss_weight=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', total=len(dataloader))

    if loss_weight is None:
        loss_weight = {'actions': 1.0, 'looks': 1.0, 'crosses': 1.0}

    for batch_idx, (images, motions, labels) in enumerate(progress_bar):
        start_time = time.time()
        images = images.to(device, non_blocking=True)
        motions = motions.to(device, non_blocking=True)
        labels = {k: v.to(device) for k, v in labels.items()}

        remap_cross_labels(labels)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, motions)

        total_batch_loss = 0.0
        loss_dict = {}
        for name in outputs:
            logits = outputs[name]
            targets = labels[name]
            head_loss = criterion[name](logits, targets)
            weighted_loss = loss_weight.get(name, 1.0) * head_loss
            total_batch_loss += weighted_loss
            loss_dict[name] = head_loss.item()
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item()

        del outputs, logits, targets, head_loss, weighted_loss

        end_time = time.time()
        batch_time = end_time - start_time

        progress_bar.set_postfix({'loss':f'{total_batch_loss.item():.4f}'})
        if (batch_idx+1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            tqdm.write(f"Batch {batch_idx}/{len(dataloader)}, Loss: {total_batch_loss.item():.4f}, Time per batch: {batch_time:.3f} sec")

    progress_bar.close()
    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Average chunk Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Returns:
      - loss_sum: float (sum of per-sample losses across the dataloader)
      - n_samples: int (total number of samples seen)
      - correct_counts: dict mapping head -> number of correct predictions (ints)
    """
    model.eval()
    loss_sum = 0.0
    correct = {}
    total = {}
    samples = 0

    with torch.no_grad():
        for images, motions, labels in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            motions = motions.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images, motions)

            # accumulate loss as sum over samples (handles criterion reduction='mean')
            batch_loss = 0.0
            for name in outputs:  # 'actions', 'looks', 'crosses'
                logits = outputs[name]
                targets = labels[name]
                loss_i = criterion[name](logits, targets)
                # convert mean loss to sum
                batch_loss += loss_i.item() * batch_size

                _, preds = torch.max(logits, 1)
                correct[name] = correct.get(name, 0) + (preds == targets).sum().item()
                total[name] = total.get(name, 0) + targets.size(0)

            samples += batch_size
            loss_sum += batch_loss

    if samples == 0:
        return 0.0, 0, {}

    # Note: return raw correct counts (not per-chunk accuracies)
    return loss_sum, samples, correct

# Define PTChunkDataset once
class PTChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data): 
        self._items = []
        for item in data:
            self._items.append(item)
    def __len__(self): return len(self._items)
    def __getitem__(self, idx): 
        item = self._items[idx]
        return item
    

def finetune(model, enable_finetune=False):
    if not enable_finetune:
        return
    for name, param in model.named_parameters():
        param.requires_grad = False
        if ('cross_attention' in name) or ('classifier' in name) or ('cross_attn' in name):
            param.requires_grad = True

def gather_chunks(folders):
    if isinstance(folders, str):
        folders = [folders]
    all_files = []
    for folder in folders:
        chunk_files = sorted([os.path.join(folder, f) 
                            for f in os.listdir(folder) 
                            if f.endswith('.pt')])
        all_files.extend(chunk_files)
    
    return all_files

def wait_for_memory(threshold=96, interval=1):
    while psutil.virtual_memory().percent > threshold:
        print(f"RAM at {psutil.virtual_memory().percent:.1f}%, waiting...")
        time.sleep(interval)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datetime_str = datetime.now().strftime("%m%d_%H%M")
    log_file = f'training_log/training_log_{datetime_str}.csv'

    os.makedirs('training_log', exist_ok=True)
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Epoch',
            'Avg Train Loss',
            'Actions Acc',
            'Looks Acc',
            'Crosses Acc',
            'Val Loss',
            'Overall Val Acc'
        ])

    print(f'Logging training progress to {log_file}')

    # Configuration
    embedding_dim = 128
    learning_rate = 1e-5
    batch_size = 8
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    num_epochs = 10
    num_workers = 0
    # Number of prediction classes per head
    num_classes_dict = {
            'actions': 2,
            'looks': 2,
            'crosses': 2
        }
    # Per-head Loss weighting
    loss_weight = {'actions': 1.0, 'looks': 0.6, 'crosses': 1.2}

    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    best_val_loss = float('inf')

    model = EnsembleModel(
        motion_enc=MotionEncoder(**motion_enc_args),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict)
    ).to(device)

    # Load model
    checkpoint_path = 'outputs/best_model_epoch.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f'Checkpoint {checkpoint_path} not found. Starting from scratch.')

    finetune(model, enable_finetune=False)

    # Class weighting for 'looks' labels. Criterion, optimizer, scheduler
    criterion = {}
    for name in num_classes_dict:
        if name == 'looks':
            criterion[name] = nn.CrossEntropyLoss(weight=class_weight(45000, 5000, device))
        elif name == 'crosses':
            criterion[name] = nn.CrossEntropyLoss(weight=class_weight(40000, 10000, device))
        else:
            criterion[name] = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=0.0001, threshold_mode='rel'
    )

    os.makedirs('outputs', exist_ok=True)

    # Asynchronous chunk loading
    def mp_async_load(idx, path, queue):
        try:
            data = torch.load(path, map_location='cpu')
            queue.put((idx, 'ok', data))
        except Exception as e:
            queue.put((idx, 'err', str(e)))

    # --- Training loop ---
    train_chunk_folder = ['preprocessed_train_base', 'preprocessed_train_augmented']
    val_chunk_folder = 'preprocessed_val_base'
    train_chunk_files = gather_chunks(train_chunk_folder)
    val_chunk_files = gather_chunks(val_chunk_folder)

    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    queue = mp.Queue(maxsize=3)
    processes = {}
    results = {}

    print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_chunk_files)
        epoch_loss = []

        preload = min(2, len(train_chunk_files))
        for i in range(preload):
            wait_for_memory(threshold=96, interval=1)
            p = mp.Process(target=mp_async_load, args=(i, train_chunk_files[i], queue))
            p.start()
            processes[i] = p

        for chunk_idx, chunk_path in enumerate(train_chunk_files):
            # Collect queue results until desired chunk
            try:
                while chunk_idx not in results:
                    idx, status, payload = queue.get(timeout=300)  # seconds (tunable)
                    results[idx] = (status, payload)
            except Empty:
                print(f"Timeout waiting for chunk {chunk_idx} — terminating associated process")
                proc = processes.pop(chunk_idx, None)
                if proc is not None:
                    proc.terminate()
                    proc.join()
                continue
            status, payload = results.pop(chunk_idx)

            # Join the process that produce this chunk
            proc = processes.pop(chunk_idx, None)
            if proc is not None:
                proc.join()
            
            if status == 'err':
                print(f'Failed to preload {chunk_path}: {payload}')
                continue
            current_data = payload

            del payload

            dataset = PTChunkDataset(current_data)
            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=False,
                persistent_workers=False,
            )
            if num_workers > 0:
                loader_kwargs['prefetch_factor'] = 2

            loader = DataLoader(dataset, **loader_kwargs)
            print(f"\n[Chunk {chunk_idx + 1}/{len(train_chunk_files)}] Training {len(loader)} batches from {chunk_path}")
            print(f"→ Training {len(loader)} batches in this chunk")
            avg_loss = train_one_chunk(model, loader, criterion, optimizer, device, loss_weight=loss_weight)
            epoch_loss.append(avg_loss)

            del current_data, dataset, loader
            del payload
            torch.cuda.empty_cache()
            trash = gc.collect()
            print(f"Unreachable trash: {trash}")

            next_idx = chunk_idx + preload
            if next_idx < len(train_chunk_files):
                wait_for_memory(threshold=96, interval=1)
                p = mp.Process(target=mp_async_load, args=(next_idx, train_chunk_files[next_idx], queue))
                p.start()
                processes[next_idx] = p
            
            preload = min(preload, 2)
        
        # collect any remaining queue items and join remaining processes
        remaining = len(processes)
        for _ in range(remaining):
            try:
                idx, status, payload = queue.get(timeout=2)
                # discard or free payload immediately
                if status == 'ok':
                    del payload
                results.pop(idx, None)
            except Exception:
                pass

        for idx, proc in list(processes.items()):
            proc.join()
            processes.pop(idx, None)

        # final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # ---- end of chunks ----
        if len(epoch_loss) == 0:
            avg_epoch_loss = float('nan')
        else:
            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # ---- validation ----
        total_val_loss_sum = 0.0
        total_val_samples = 0
        total_correct_counts = {}  # head -> total correct across all val chunks
        total_label_counts = {}    # head -> total labels across all val chunks

        for chunk_path in val_chunk_files:
            print(f"Loading validation chunk {chunk_path}")
            val_data = torch.load(chunk_path, map_location='cpu')
            val_data = filter_irrelevant(val_data)

            if not val_data:
                print(f"Validation chunk {chunk_path} is empty, skipping.")
                continue

            val_dataset = PTChunkDataset(val_data)
            del val_data
            gc.collect()

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=False
            )

            chunk_loss_sum, chunk_n, chunk_corrects = validate_one_epoch(model, val_loader, criterion, device)

            total_val_loss_sum += chunk_loss_sum
            total_val_samples += chunk_n

            # aggregate correct counts and per-head totals
            for head, corr_count in chunk_corrects.items():
                total_correct_counts[head] = total_correct_counts.get(head, 0) + int(corr_count)
                # for per-head totals, assume chunk_n is the number of samples for that head
                # (we used labels.size(0) in validate_one_epoch, which is same as batch_size)
                total_label_counts[head] = total_label_counts.get(head, 0) + chunk_n

            del val_dataset, val_loader, chunk_corrects
            gc.collect()
            torch.cuda.empty_cache()

        if total_val_samples == 0:
            raise RuntimeError("No validation samples found!")

        # final averaged loss (per-sample)
        val_loss = total_val_loss_sum / total_val_samples

        # compute per-head accuracies and overall accuracy
        val_metric = {}
        for head in total_correct_counts:
            val_metric[head] = total_correct_counts[head] / total_label_counts[head] if total_label_counts[head] > 0 else 0.0

        overall_acc = sum(total_correct_counts.values()) / sum(total_label_counts.values()) if sum(total_label_counts.values()) > 0 else 0.0
        val_metric['overall'] = overall_acc

        scheduler.step(val_loss)


        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                round(avg_epoch_loss, 4),
                round(val_metric.get('actions', 0.0), 4),
                round(val_metric.get('looks', 0.0), 4),
                round(val_metric.get('crosses', 0.0), 4),
                round(val_loss, 4),
                round(val_metric.get('overall', 0.0), 4)
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), f'outputs/best_model_epoch{epoch+1}_{datetime_str}.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Saving final model and stopping.")
            torch.save(model.state_dict(), f'outputs/final_model_epoch{epoch+1}_{datetime_str}.pth')
            break

if __name__ == "__main__":
    main()