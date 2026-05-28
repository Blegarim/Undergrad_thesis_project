import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model
from scripts.lmdb_dataset import LMDBChunkDataset
from scripts.model_utils import get_model, model_forward
from scripts.train_utils import (
    collate_fn, EarlyStopping, remap_cross_labels,
    gather_chunks, wait_for_memory, mp_async_load
)

import gc
import random
import csv
import psutil
from collections import Counter
import multiprocessing as mp
import lmdb, pickle
from queue import Empty
from tqdm import tqdm
from datetime import datetime

'''
Training script for the PIE dataset using an Ensemble Model with Temporal ConvNet-GRU-Attention, Hierarchical Vision Transformer and Cross Attention.
'''

def compute_class_weights_from_lmdb(lmdb_paths, device):
    """
    Compute per-task class weights from LMDB metadata.
    Uses inverse frequency weighting: weight_i = total / (n_classes * count_i)
    This upweights minority classes during loss computation.
    """
    counts = {task: [0, 0] for task in ['actions', 'looks', 'crosses']}
    
    for path in lmdb_paths:
        env = lmdb.open(path, readonly=True, lock=False)
        try:
            with env.begin(write=False) as txn:
                for key, value in txn.cursor():
                    key_str = key.decode()
                    if key_str.endswith('_meta'):
                        meta = pickle.loads(value)
                        for task in counts:
                            label = int(meta.get(task, 0))
                            if task == 'crosses':
                                # Mirror remap_cross_labels: raw {-1, 0, 1} → {0, 1}
                                label = max(0, min(1, label))
                            if label in [0, 1]:
                                counts[task][label] += 1
        finally:
            env.close()
    
    weights = {}
    for task, cnt in counts.items():
        total = sum(cnt)
        if total > 0:
            cnt0, cnt1 = cnt[0], cnt[1]
            weights[task] = torch.tensor([
                total / (2 * max(cnt0, 1)),
                total / (2 * max(cnt1, 1))
            ], dtype=torch.float32, device=device)
        else:
            weights[task] = torch.tensor([1.0, 1.0], device=device)
    
    return weights

def _inverse_class_weights(counts):
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for k, v in counts.items():
        if v == 0:
            weights[k] = 0.0
        else:
            weights[k] = total / (n_classes * v)
    return weights

def build_sampler_weights(lmdb_path, seq_ids, cross_pow=1.0, action_pow=0.5, look_pow=0.5, min_weight=1e-6):
    label_rows = []
    counts = {
        "actions": Counter(),
        "looks": Counter(),
        "crosses": Counter(),
    }

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn:
            for seq_id in seq_ids:
                meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
                actions = int(meta["actions"])
                looks = int(meta["looks"])
                crosses = int(meta["crosses"])
                if crosses < 0:
                    crosses = 0
                label_rows.append((actions, looks, crosses))
                counts["actions"][actions] += 1
                counts["looks"][looks] += 1
                counts["crosses"][crosses] += 1
    finally:
        env.close()

    action_w = _inverse_class_weights(counts["actions"])
    look_w = _inverse_class_weights(counts["looks"])
    cross_w = _inverse_class_weights(counts["crosses"])

    weights = []
    for actions, looks, crosses in label_rows:
        weight = max(min_weight, cross_w.get(crosses, min_weight)) ** cross_pow
        if action_pow > 0:
            weight *= max(min_weight, action_w.get(actions, min_weight)) ** action_pow
        if look_pow > 0:
            weight *= max(min_weight, look_w.get(looks, min_weight)) ** look_pow
        weights.append(weight)

    return weights, counts

def train_one_chunk(model, dataloader, criterion, optimizer, device, model_type, loss_weight=None, scaler=None, use_amp=False, use_pin_memory=False):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', total=len(dataloader))

    if loss_weight is None:
        loss_weight = {'actions': 1.0, 'looks': 1.0, 'crosses': 1.0}

    for (images_tight, images_context, motions, labels) in progress_bar:
        images_tight = images_tight.to(device, non_blocking=use_pin_memory)
        images_context = images_context.to(device, non_blocking=use_pin_memory)
        motions = motions.to(device, non_blocking=use_pin_memory)
        labels = {k: v.to(device, non_blocking=use_pin_memory).long() for k, v in labels.items()}

        remap_cross_labels(labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model_forward(model, model_type, images_tight, images_context, motions)

            total_batch_loss = torch.tensor(0.0, device=device)
            for name in ["actions", "looks", "crosses"]:
                if name == "crosses":
                    logits = outputs["crosses_frame"]
                else:
                    logits = outputs[name]

                targets = labels[name]
                head_loss = criterion[name](logits.float(), targets)
                total_batch_loss = total_batch_loss + loss_weight.get(name, 1.0) * head_loss

        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += total_batch_loss.item()

        progress_bar.set_postfix({'loss':f'{total_batch_loss.item():.4f}'})

    progress_bar.close()
    n_batches = len(dataloader)
    if n_batches == 0:
        return 0.0, 0
    tqdm.write(f"Average chunk Loss: {total_loss / n_batches:.4f}")
    torch.cuda.empty_cache()
    return total_loss, n_batches

def validate_one_epoch(model, dataloader, criterion, device, model_type, loss_weight=None, use_amp=False, use_pin_memory=False):
    """
    Returns:
      - loss_sum: float (sum of per-sample losses across the dataloader)
      - n_samples: int (total number of samples seen)
      - correct_counts: dict mapping head -> number of correct predictions (ints)
      - all_preds: dict mapping head -> list of predictions (for F1 computation)
      - all_targets: dict mapping head -> list of targets (for F1 computation)
    """
    model.eval()
    if loss_weight is None:
        loss_weight = {'actions': 1.0, 'looks': 1.0, 'crosses': 1.0}
    loss_sum = 0.0
    correct = {}
    all_preds = {name: [] for name in ["actions", "looks", "crosses"]}
    all_targets = {name: [] for name in ["actions", "looks", "crosses"]}
    samples = 0

    with torch.inference_mode():
        for images_tight, images_context, motions, labels in dataloader:
            batch_size = images_tight.size(0)
            images_tight = images_tight.to(device, non_blocking=use_pin_memory)
            images_context = images_context.to(device, non_blocking=use_pin_memory)
            motions = motions.to(device, non_blocking=use_pin_memory)
            labels = {k: v.to(device, non_blocking=use_pin_memory).long() for k, v in labels.items()}

            remap_cross_labels(labels)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model_forward(model, model_type, images_tight, images_context, motions)

            # accumulate loss as sum over samples (handles criterion reduction='mean')
            batch_loss = 0.0
            for name in ["actions", "looks", "crosses"]:
                if name == "crosses":
                    logits = outputs["crosses_frame"]
                else:
                    logits = outputs[name]
                if use_amp:
                    logits = logits.float()
                targets = labels[name]
                loss_i = criterion[name](logits, targets)
                # convert mean loss to sum; mirror train_one_chunk's per-head weighting
                batch_loss += loss_weight.get(name, 1.0) * loss_i.item() * batch_size

                _, preds = torch.max(logits, 1)
                correct[name] = correct.get(name, 0) + (preds == targets).sum().item()

                all_preds[name].extend(preds.cpu().numpy().tolist())
                all_targets[name].extend(targets.cpu().numpy().tolist())

            samples += batch_size
            loss_sum += batch_loss

    if samples == 0:
        return 0.0, 0, {}, {}, {}

    # Note: return raw correct counts and predictions (not per-chunk accuracies)
    return loss_sum, samples, correct, all_preds, all_targets
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train pedestrian behavior prediction model')
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['motion_only', 'visual_only', 'vanilla_concat', 'full'],
                        help='Model type for ablation study')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    use_amp = device.type == "cuda"
    use_pin_memory = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

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
            'Actions F1',
            'Looks F1',
            'Crosses F1',
            'Macro F1',
            'Val Loss',
            'Overall Val Acc'
        ])

    print(f'Logging training progress to {log_file}')

    # Configuration
    embedding_dim = get_unified_dim_model()
    learning_rate = 1e-4
    batch_size = 4
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    num_epochs = 30
    num_workers = 6
    num_classes_dict = {
            'actions': 2,
            'looks': 2,
            'crosses': 2
        }
    loss_weight = {'actions': 0.8, 'looks': 0.8, 'crosses': 1.2}
    use_weighted_sampler = True 
    sampler_powers = {"crosses": 1.5, "actions": 0.3, "looks": 0.7}
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_val_loss = float('inf')

    # Initialize base components
    motion_enc = MotionEncoder(**motion_enc_args)
    vit = ViT_Hierarchical(**vit_args)
    
    # Get model based on type selection
    model = get_model(args.model_type, motion_enc, vit, embedding_dim, num_classes_dict).to(device)

    # Materialize lazily-built parameters BEFORE checkpoint load + optimizer construction.
    # Why: ViT_Hierarchical's global-window blocks (window_size=None) defer their
    # relative_position_bias_table to the first forward call. If the optimizer is
    # built first, those late-created parameters never receive gradient updates,
    # and load_state_dict(strict=False) drops them as "unexpected" on resume.
    with torch.no_grad():
        model.eval()
        dummy_tight = torch.zeros(1, 2, 3, 128, 128, device=device)
        dummy_context = torch.zeros(1, 2, 3, 224, 224, device=device)
        dummy_motions = torch.zeros(1, 2, 8, device=device)
        model_forward(model, args.model_type, dummy_tight, dummy_context, dummy_motions)
        model.train()

    # Load model
    checkpoint_path = 'best_model_outputs/best_model_epoch.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        state_dict = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    else:
        print(f'Checkpoint {checkpoint_path} not found. Starting from scratch.')

    train_chunk_folder = ['preprocessed_train', 'preprocessed_train_aug']
    val_chunk_folder = 'preprocessed_val'
    train_chunk_files = gather_chunks(train_chunk_folder)
    val_chunk_files = gather_chunks(val_chunk_folder)

    print("Computing class weights from training data for severe imbalance handling...")
    class_weights = compute_class_weights_from_lmdb(train_chunk_files, device)
    for task in ['actions', 'looks', 'crosses']:
        w = class_weights[task]
        print(f"  {task}: class_0 weight={w[0].item():.2f}, class_1 weight={w[1].item():.2f}")

    criterion = {
        "actions": nn.CrossEntropyLoss(weight=class_weights['actions']),
        "looks": nn.CrossEntropyLoss(weight=class_weights['looks']),
        "crosses": nn.CrossEntropyLoss(weight=class_weights['crosses'])
    }
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel'
    )

    os.makedirs('model_outputs', exist_ok=True)
    os.makedirs('best_model_outputs', exist_ok=True)

    transform_tight = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_context = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Training loop ---
    weight_cache = {}
    queue = mp.Queue(maxsize=3)
    processes = {}
    results = {}

    print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_chunk_files)
        epoch_loss_sum = 0.0
        epoch_n_batches = 0

        preload = min(3, len(train_chunk_files))
        for i in range(preload):
            wait_for_memory(threshold=96, interval=1)
            p = mp.Process(target=mp_async_load, args=(i, train_chunk_files[i], queue))
            p.start()
            processes[i] = p

        try:
            for chunk_idx, chunk_path in enumerate(train_chunk_files):
                # Collect queue results until desired chunk
                try:
                    while chunk_idx not in results:
                        idx, status, payload = queue.get(timeout=300)
                        results[idx] = (status, payload)
                except Empty:
                    print(f"Timeout waiting for chunk {chunk_idx} — terminating associated process")
                    proc = processes.pop(chunk_idx, None)
                    if proc is not None:
                        proc.terminate()
                        proc.join()
                    continue
                status, payload = results.pop(chunk_idx)

                proc = processes.pop(chunk_idx, None)
                if proc is not None:
                    proc.join()

                if status == 'err':
                    print(f'Failed to preload {chunk_path}: {payload}')
                    continue
                lmdb_path = payload

                del payload

                dataset = LMDBChunkDataset(lmdb_path, transform_tight=transform_tight, transform_context=transform_context)
                loader_kwargs = dict(
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=use_pin_memory,
                    persistent_workers=False,
                )
                if num_workers > 0:
                    loader_kwargs['prefetch_factor'] = 2

                if use_weighted_sampler:
                    cached = weight_cache.get(lmdb_path)
                    if cached is None:
                        weights, counts = build_sampler_weights(
                            lmdb_path,
                            dataset.seq_ids,
                            cross_pow=sampler_powers["crosses"],
                            action_pow=sampler_powers["actions"],
                            look_pow=sampler_powers["looks"],
                        )
                        weight_cache[lmdb_path] = (weights, counts)
                    else:
                        weights, counts = cached

                    sampler = WeightedRandomSampler(
                        weights=torch.DoubleTensor(weights),
                        num_samples=len(weights),
                        replacement=True,
                    )
                    loader_kwargs["sampler"] = sampler
                    loader_kwargs["shuffle"] = False
                    print(
                        "Sampler counts: "
                        f"actions={dict(counts['actions'])} "
                        f"looks={dict(counts['looks'])} "
                        f"crosses={dict(counts['crosses'])} "
                        f"powers={sampler_powers}"
                    )

                loader = DataLoader(dataset, **loader_kwargs)
                print(f"\n[Chunk {chunk_idx + 1}/{len(train_chunk_files)}] {len(loader)} batches from {chunk_path}")

                chunk_loss_sum, chunk_n_batches = train_one_chunk(
                    model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    args.model_type,
                    loss_weight=loss_weight,
                    scaler=scaler,
                    use_amp=use_amp,
                    use_pin_memory=use_pin_memory,
                )
                epoch_loss_sum += chunk_loss_sum
                epoch_n_batches += chunk_n_batches

                del lmdb_path, dataset, loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                next_idx = chunk_idx + preload
                if next_idx < len(train_chunk_files):
                    wait_for_memory(threshold=96, interval=1)
                    p = mp.Process(target=mp_async_load, args=(next_idx, train_chunk_files[next_idx], queue))
                    p.start()
                    processes[next_idx] = p

        finally:
            for idx, proc in list(processes.items()):
                proc.terminate()
                proc.join()
            processes.clear()
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Exception:
                    break

        # final cleanup — clear results to prevent stale entries bleeding into the next epoch
        results.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # ---- end of chunks ----
        if epoch_n_batches == 0:
            avg_epoch_loss = float('nan')
        else:
            avg_epoch_loss = epoch_loss_sum / epoch_n_batches
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # Save model with model type suffix
        model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
        torch.save(model.state_dict(), f'model_outputs/checkpoint_{datetime_str}{model_suffix}.pth')

        # ---- validation ----
        total_val_loss_sum = 0.0
        total_val_samples = 0
        total_correct_counts = {}  # head -> total correct across all val chunks
        total_label_counts = {}    # head -> total labels across all val chunks
        val_all_preds = {name: [] for name in ["actions", "looks", "crosses"]}
        val_all_targets = {name: [] for name in ["actions", "looks", "crosses"]}

        for chunk_path in val_chunk_files:
            print(f"Loading validation chunk {chunk_path}")
            val_dataset = LMDBChunkDataset(
                chunk_path,
                transform_tight=transform_tight,
                transform_context=transform_context
            )

            val_loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=use_pin_memory,
            )
            if num_workers > 0:
                val_loader_kwargs['prefetch_factor'] = 2
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)

            chunk_loss_sum, chunk_n, chunk_corrects, chunk_preds, chunk_targets = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.model_type,
                loss_weight=loss_weight,
                use_amp=use_amp,
                use_pin_memory=use_pin_memory,
            )

            total_val_loss_sum += chunk_loss_sum
            total_val_samples += chunk_n

            # aggregate correct counts and per-head totals
            for head, corr_count in chunk_corrects.items():
                total_correct_counts[head] = total_correct_counts.get(head, 0) + int(corr_count)
                # for per-head totals, assume chunk_n is the number of samples for that head
                total_label_counts[head] = total_label_counts.get(head, 0) + chunk_n
            
            # aggregate predictions for F1 computation
            for head in ["actions", "looks", "crosses"]:
                val_all_preds[head].extend(chunk_preds.get(head, []))
                val_all_targets[head].extend(chunk_targets.get(head, []))

            del val_dataset, val_loader, chunk_corrects, chunk_preds, chunk_targets
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if total_val_samples == 0:
            raise RuntimeError("No validation samples found!")

        # final averaged loss (per-sample)
        val_loss = total_val_loss_sum / total_val_samples

        # compute per-head accuracies and overall accuracy
        val_metric = {}
        for head in total_correct_counts:
            val_metric[head] = total_correct_counts[head] / total_label_counts[head] if total_label_counts[head] > 0 else 0.0
        
        # compute F1, precision, recall for each head
        for head in ["actions", "looks", "crosses"]:
            preds = val_all_preds[head]
            targets = val_all_targets[head]
            if len(set(targets)) > 1:
                val_metric[f'{head}_f1'] = f1_score(targets, preds, average='binary', zero_division=0)
                val_metric[f'{head}_precision'] = precision_score(targets, preds, average='binary', zero_division=0)
                val_metric[f'{head}_recall'] = recall_score(targets, preds, average='binary', zero_division=0)
            else:
                val_metric[f'{head}_f1'] = 0.0
                val_metric[f'{head}_precision'] = 0.0
                val_metric[f'{head}_recall'] = 0.0
        
        # compute macro F1
        macro_f1 = (val_metric.get('actions_f1', 0) + val_metric.get('looks_f1', 0) + val_metric.get('crosses_f1', 0)) / 3

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
                round(val_metric.get('actions_f1', 0.0), 4),
                round(val_metric.get('looks_f1', 0.0), 4),
                round(val_metric.get('crosses_f1', 0.0), 4),
                round(macro_f1, 4),
                round(val_loss, 4),
                round(val_metric.get('overall', 0.0), 4)
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
            torch.save(model.state_dict(), f'best_model_outputs/best_model_epoch{epoch+1}_{datetime_str}{model_suffix}.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Saving final model and stopping.")
            model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
            torch.save(model.state_dict(), f'model_outputs/final_model_epoch{epoch+1}_{datetime_str}{model_suffix}.pth')
            break

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
