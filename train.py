import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from models.AblationModels import MotionOnlyModel, VisualOnlyModel, VanillaConcatModel
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model
from scripts.lmdb_dataset import LMDBChunkDataset

import time
import gc
import random
import csv
import psutil
from collections import Counter
import multiprocessing as mp
import multiprocessing.queues as mpq
import subprocess
import lmdb, pickle
from queue import Empty
from tqdm import tqdm
from datetime import datetime

'''
Training script for the PIE dataset using an Ensemble Model with Temporal ConvNet-GRU-Attention, Hierarchical Vision Transformer and Cross Attention.
'''

def collate_fn(batch):
    images_tight = torch.stack([item['images_tight'] for item in batch], dim=0)
    images_context = torch.stack([item['images_context'] for item in batch], dim=0)
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :8]
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
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model_forward(model, model_type, images_tight, images_context, motions)

        total_batch_loss = 0.0
        for name in ["actions", "looks", "crosses"]:
            if name == "crosses":
                if model_type in ['full', 'vanilla_concat'] and hasattr(model, 'cross_attention') and model.cross_attention.use_frame_crosses:
                    logits = outputs["crosses_frame"]
                else:
                    logits = outputs["crosses_pooled"]
            else:
                logits = outputs[name]

            targets = labels[name]
            head_loss = criterion[name](logits, targets)
            total_batch_loss += loss_weight.get(name, 1.0) * head_loss

        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            optimizer.step()
        total_loss += total_batch_loss.item()

        del outputs, logits, targets, head_loss

        progress_bar.set_postfix({'loss':f'{total_batch_loss.item():.4f}'})

    progress_bar.close()
    if len(dataloader) == 0:
        return float('nan')  
    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Average chunk Loss: {avg_loss:.4f}")
    torch.cuda.empty_cache()
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device, model_type, use_amp=False, use_pin_memory=False):
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

    with torch.inference_mode():
        for images_tight, images_context, motions, labels in dataloader:
            batch_size = images_tight.size(0)
            images_tight = images_tight.to(device, non_blocking=use_pin_memory)
            images_context = images_context.to(device, non_blocking=use_pin_memory)
            motions = motions.to(device, non_blocking=use_pin_memory)
            labels = {k: v.to(device, non_blocking=use_pin_memory).long() for k, v in labels.items()}

            remap_cross_labels(labels)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model_forward(model, model_type, images_tight, images_context, motions)

            # accumulate loss as sum over samples (handles criterion reduction='mean')
            batch_loss = 0.0
            for name in ["actions", "looks", "crosses"]: 
                if name == "crosses":
                    if model_type in ['full', 'vanilla_concat'] and hasattr(model, 'cross_attention') and model.cross_attention.use_frame_crosses:
                        logits = outputs["crosses_frame"]
                    else:
                        logits = outputs["crosses_pooled"]
                else:
                    logits = outputs[name]
                if use_amp:
                    logits = logits.float()
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
                            if f.endswith('.lmdb')])
        all_files.extend(chunk_files)
    
    return all_files

def wait_for_memory(threshold=96, interval=1):
    while psutil.virtual_memory().percent > threshold:
        print(f"RAM at {psutil.virtual_memory().percent:.1f}%, waiting...")
        time.sleep(interval)

def mp_async_load(idx, path, queue):
    """
    Warm LMDB chunk file (light read) in a background process, then return the path.
    For .pt files this would torch.load; for LMDB we just open & read a tiny bit
    to encourage the OS to cache the file, then pass back the path string.
    """
    try:
        # Quick warm: open LMDB and read one _meta key if available
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin(write=False) as txn:
            # iterate until we find a meta key, then break
            cursor = txn.cursor()
            for key, _ in cursor:
                key_s = key.decode()
                if key_s.endswith("_meta"):
                    # read one meta to warm
                    _ = txn.get(key)
                    break
        env.close()
        # Return the path string as payload; parent will instantiate LMDBChunkDataset(path)
        queue.put((idx, 'ok', path))
    except Exception as e:
        queue.put((idx, 'err', str(e)))

def get_hdd_temp(dev="/dev/sda"):
    try:
        out = subprocess.check_output(["hdddtemp", dev]).decode()
        return int(out.split(":")[2].strip().split("°")[0])
    except Exception:
        return None

def get_model(model_type, motion_enc, vit, d_model, num_classes_dict):
    """
    Get specified model for ablation study.
    
    Args:
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', or 'full'
        motion_enc: MotionEncoder instance
        vit: ViT_Hierarchical instance  
        d_model: model dimension
        num_classes_dict: dictionary of class counts
    
    Returns:
        Model instance
    """
    if model_type == 'motion_only':
        return MotionOnlyModel(
            motion_enc=motion_enc,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
    elif model_type == 'visual_only':
        return VisualOnlyModel(
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
    elif model_type == 'vanilla_concat':
        return VanillaConcatModel(
            motion_enc=motion_enc,
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
    elif model_type == 'full':
        # Original full model with cross-attention
        cross_attention = CrossAttentionModule(
            d_model=d_model,
            num_heads=4,
            num_classes_dict=num_classes_dict,
            use_frame_crosses=True,
            frame_pool="logsumexp",
        )
        return EnsembleModel(
            motion_enc=motion_enc,
            vit=vit,
            cross_attention=cross_attention,
            d_model=d_model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: 'motion_only', 'visual_only', 'vanilla_concat', 'full'")

def model_forward(model, model_type, images_tight, images_context, motions):
    """
    Forward pass wrapper for different model types.
    
    Args:
        model: model instance
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', 'full'
        images_tight: [B, T, C, H, W]
        images_context: [B, T, C, H, W] 
        motions: [B, T, motion_dim]
    
    Returns:
        logits dict
    """
    if model_type == 'motion_only':
        # Need to extract motion features first
        motion_feats = model.motion_enc(motions, images_tight)
        logits = model(motion_feats)
    elif model_type == 'visual_only':
        logits = model(images_context)
    else:  # 'vanilla_concat' or 'full'
        logits = model(images_tight, images_context, motions)
    
    return logits

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
    num_workers = 4
    num_classes_dict = {
            'actions': 2,
            'looks': 2,
            'crosses': 2
        }
    loss_weight = {'actions': 0.8, 'looks': 0.8, 'crosses': 1.2}
    use_weighted_sampler = True 
    sampler_powers = {"crosses": 1, "actions": 0.5, "looks": 0.5}
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_val_loss = float('inf')

    # Initialize base components
    motion_enc = MotionEncoder(**motion_enc_args)
    vit = ViT_Hierarchical(**vit_args)
    
    # Get model based on type selection
    model = get_model(args.model_type, motion_enc, vit, embedding_dim, num_classes_dict).to(device)

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

    finetune(model, enable_finetune=False)

    criterion = {
        "actions": nn.CrossEntropyLoss(),
        "looks": nn.CrossEntropyLoss(),
        "crosses": nn.CrossEntropyLoss()
    }
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel'
    )

    os.makedirs('model_outputs', exist_ok=True)
    os.makedirs('best_model_outputs', exist_ok=True)

    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Training loop ---
    weight_cache = {}
    train_chunk_folder = ['preprocessed_train', 'preprocessed_train_aug']
    val_chunk_folder = 'preprocessed_val'
    train_chunk_files = gather_chunks(train_chunk_folder)
    val_chunk_files = gather_chunks(val_chunk_folder)

    queue = mp.Queue(maxsize=3)
    processes = {}
    results = {}

    print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_chunk_files)
        epoch_loss = []

        preload = min(3, len(train_chunk_files))
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
            lmdb_path = payload

            del payload

            dataset = LMDBChunkDataset(lmdb_path, transform_tight=base_transforms, transform_context=base_transforms)
            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=use_pin_memory,
                persistent_workers=False,
            )
            if num_workers > 0:
                loader_kwargs['prefetch_factor'] = 1

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
            print(f"\n[Chunk {chunk_idx + 1}/{len(train_chunk_files)}] Training {len(loader)} batches from {chunk_path}")
            print(f"→ Training {len(loader)} batches in this chunk")
            
            avg_loss = train_one_chunk(
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
            epoch_loss.append(avg_loss)

            del lmdb_path, dataset, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            trash = gc.collect()
            print(f"Unreachable trash: {trash}")

            # Additional memory cleanup - removed duplicate empty_cache

            next_idx = chunk_idx + preload
            if next_idx < len(train_chunk_files):
                wait_for_memory(threshold=96, interval=1)
                p = mp.Process(target=mp_async_load, args=(next_idx, train_chunk_files[next_idx], queue))
                p.start()
                processes[next_idx] = p
            
            preload = min(preload, 3)
        
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

        # Save model with model type suffix
        model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
        torch.save(model.state_dict(), f'model_outputs/checkpoint_{datetime_str}{model_suffix}.pth')

        # ---- validation ----
        total_val_loss_sum = 0.0
        total_val_samples = 0
        total_correct_counts = {}  # head -> total correct across all val chunks
        total_label_counts = {}    # head -> total labels across all val chunks

        for chunk_path in val_chunk_files:
            print(f"Loading validation chunk {chunk_path}")
            val_dataset = LMDBChunkDataset(
                chunk_path,
                transform_tight=base_transforms,   
                transform_context=base_transforms
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=use_pin_memory
            )

            chunk_loss_sum, chunk_n, chunk_corrects = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.model_type,
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

            del val_dataset, val_loader, chunk_corrects
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
            model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
            torch.save(model.state_dict(), f'best_model_outputs/best_model_epoch{epoch+1}_{datetime_str}{model_suffix}.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Saving final model and stopping.")
            model_suffix = f"_{args.model_type}" if args.model_type != 'full' else ""
            torch.save(model.state_dict(), f'model_outputs/final_model_epoch{epoch+1}_{datetime_str}{model_suffix}.pth')
            break
        
        # temp = get_hdd_temp()
        # if temp and temp >= 50:
        #     rest = 180
        # elif temp and temp >= 40:
        #     rest = 120
        # else:
        #     rest = 1
        # print(f"HDD at {temp}, resting for {rest}...")
        time.sleep(1)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
