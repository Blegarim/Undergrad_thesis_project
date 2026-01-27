"""
Enhanced training script with advanced class imbalance handling.
This script extends the original train.py with toggleable imbalance strategies.
Usage: python train_enhanced.py --preset recommended --enable_advanced True
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config
from scripts.lmdb_dataset import LMDBChunkDataset

# Import new imbalance handling components
from class_imbalance_strategies import (
    FocalLoss, ClassBalancedFocalLoss, DynamicLossWeighting, 
    ThresholdOptimizer, HardNegativeMining
)
from imbalance_config import (
    get_training_config, get_preset_config, validate_config,
    apply_imbalance_strategies_to_training
)

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
import argparse

# Import original training functions
from train import (
    collate_fn, EarlyStopping, remap_cross_labels, filter_irrelevant,
    class_weight, _inverse_class_weights, build_sampler_weights,
    train_one_chunk, validate_one_epoch, finetune, gather_chunks,
    wait_for_memory, mp_async_load, get_hdd_temp
)


def compute_f1_metrics(model, dataloader, device, use_amp=False, use_pin_memory=False):
    """
    Compute F1, precision, and recall metrics for each task.
    This is more informative than accuracy for imbalanced datasets.
    """
    model.eval()
    all_predictions = {'actions': [], 'looks': [], 'crosses': []}
    all_targets = {'actions': [], 'looks': [], 'crosses': []}
    
    with torch.inference_mode():
        for images_tight, images_context, motions, labels in dataloader:
            images_tight = images_tight.to(device, non_blocking=use_pin_memory)
            images_context = images_context.to(device, non_blocking=use_pin_memory)
            motions = motions.to(device, non_blocking=use_pin_memory)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images_tight, images_context, motions)
            
            for task in ['actions', 'looks', 'crosses']:
                if task == 'crosses':
                    if model.cross_attention.use_frame_crosses:
                        logits = outputs['crosses_frame']
                    else:
                        logits = outputs['crosses_pooled']
                else:
                    logits = outputs[task]
                
                _, preds = torch.max(logits, 1)
                all_predictions[task].extend(preds.cpu().numpy())
                all_targets[task].extend(labels[task].numpy())
    
    # Compute metrics
    metrics = {}
    for task in ['actions', 'looks', 'crosses']:
        if len(set(all_targets[task])) > 1:  # Only if both classes present
            metrics[f'{task}_f1'] = f1_score(all_targets[task], all_predictions[task], average='binary', pos_label=1)
            metrics[f'{task}_precision'] = precision_score(all_targets[task], all_predictions[task], average='binary', pos_label=1)
            metrics[f'{task}_recall'] = recall_score(all_targets[task], all_predictions[task], average='binary', pos_label=1)
            metrics[f'{task}_accuracy'] = (np.array(all_predictions[task]) == np.array(all_targets[task])).mean()
        else:
            # If only one class present, return default values
            metrics[f'{task}_f1'] = 0.0
            metrics[f'{task}_precision'] = 0.0
            metrics[f'{task}_recall'] = 0.0
            metrics[f'{task}_accuracy'] = (np.array(all_predictions[task]) == np.array(all_targets[task])).mean()
    
    return metrics


def train_one_chunk_enhanced(model, dataloader, criterion, optimizer, device, 
                           loss_weight=None, scaler=None, use_amp=False, 
                           use_pin_memory=False, dynamic_weighting=None,
                           hard_negative_mining=None):
    """
    Enhanced training function with advanced imbalance handling.
    """
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
            outputs = model(images_tight, images_context, motions)
        
        total_batch_loss = 0.0
        task_losses = {}
        
        for name in ["actions", "looks", "crosses"]:
            if name == "crosses":
                if model.cross_attention.use_frame_crosses:
                    logits = outputs["crosses_frame"]
                else:
                    logits = outputs["crosses_pooled"]
            else:
                logits = outputs[name]
            
            targets = labels[name]
            
            # Apply hard negative mining if enabled
            if hard_negative_mining is not None and name in ['looks', 'crosses']:
                hard_indices = hard_negative_mining.select_hard_negatives(logits, targets, name)
                if len(hard_indices) > 0:
                    logits = logits[hard_indices]
                    targets = targets[hard_indices]
            
            head_loss = criterion[name](logits, targets)
            task_losses[name] = head_loss
            total_batch_loss += loss_weight.get(name, 1.0) * head_loss
        
        # Apply dynamic loss weighting if enabled
        if dynamic_weighting is not None:
            total_batch_loss = dynamic_weighting.get_weighted_loss(task_losses)
        
        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            optimizer.step()
        
        total_loss += total_batch_loss.item()
        
        del outputs, logits, targets, head_loss
        
        progress_bar.set_postfix({'loss': f'{total_batch_loss.item():.4f}'})
    
    progress_bar.close()
    if len(dataloader) == 0:
        return float('nan')
    
    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Average chunk Loss: {avg_loss:.4f}")
    torch.cuda.empty_cache()
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Enhanced training with class imbalance handling')
    parser.add_argument('--preset', type=str, default='recommended',
                       choices=['conservative', 'aggressive', 'recommended'],
                       help='Configuration preset for imbalance handling')
    parser.add_argument('--enable_advanced', type=bool, default=True,
                       help='Enable advanced imbalance strategies')
    parser.add_argument('--checkpoint_path', type=str, 
                       default='best_model_outputs/best_model_epoch10_0121_2029.pth',
                       help='Path to checkpoint for loading')
    parser.add_argument('--finetune', type=bool, default=False,
                       help='Enable finetuning mode')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = device.type == "cuda"
    use_pin_memory = device.type == "cuda"
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    
    datetime_str = datetime.now().strftime("%m%d_%H%M")
    log_file = f'training_log/training_log_enhanced_{datetime_str}.csv'
    
    os.makedirs('training_log', exist_ok=True)
    
    # Get configuration
    if args.enable_advanced:
        config = get_preset_config(args.preset)
        print(f"🚀 Using {args.preset} preset with advanced imbalance handling")
    else:
        config = get_training_config()
        config['use_advanced_imbalance_handling'] = False
        print("📚 Using original training pipeline (no advanced imbalance handling)")
    
    validate_config(config)
    
    # Setup logging with enhanced metrics
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [
            'Epoch', 'Avg Train Loss',
            'Actions Acc', 'Actions F1', 'Actions Precision', 'Actions Recall',
            'Looks Acc', 'Looks F1', 'Looks Precision', 'Looks Recall',
            'Crosses Acc', 'Crosses F1', 'Crosses Precision', 'Crosses Recall',
            'Val Loss', 'Overall Val Acc', 'Macro F1'
        ]
        writer.writerow(header)
    
    print(f'📊 Logging enhanced training progress to {log_file}')
    
    # Model configuration
    embedding_dim = config['embedding_dim']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 2
    }
    
    # Initialize model
    model = EnsembleModel(
        motion_enc=MotionEncoder(**motion_enc_args),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(
            d_model=embedding_dim,
            num_heads=4,
            num_classes_dict=num_classes_dict,
            use_frame_crosses=True,
            frame_pool="logsumexp",
        )
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f'📂 Loading model from {args.checkpoint_path}')
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    else:
        print(f'⚠️ Checkpoint {args.checkpoint_path} not found. Starting from scratch.')
    
    finetune(model, enable_finetune=args.finetune)
    
    # Setup loss functions based on configuration
    if config.get('use_advanced_imbalance_handling', False):
        print("🎯 Setting up advanced loss functions...")
        
        # Create dummy datasets for loss function creation (will be updated with real data)
        dummy_datasets = {'all': []}
        
        # Apply imbalance strategies
        imbalance_setup = apply_imbalance_strategies_to_training(model, config, dummy_datasets)
        criterion = imbalance_setup['loss_functions']
        advanced_components = imbalance_setup['advanced_components']
        
        # Extract components
        dynamic_weighting = advanced_components.get('dynamic_weighting')
        hard_negative_mining = None
        if config.get('use_hard_negative_mining', False):
            hard_negative_mining = HardNegativeMining(
                ratio=config.get('hard_negative_ratio', 0.3)
            )
        
        print("✅ Advanced imbalance handling configured")
    else:
        # Use original loss functions
        criterion = {
            "actions": nn.CrossEntropyLoss(),
            "looks": nn.CrossEntropyLoss(),
            "crosses": nn.CrossEntropyLoss()
        }
        dynamic_weighting = None
        hard_negative_mining = None
        print("📚 Using standard CrossEntropyLoss")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel'
    )
    
    # Create output directories
    os.makedirs('model_outputs', exist_ok=True)
    os.makedirs('best_model_outputs', exist_ok=True)
    
    # Data transforms
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Training setup
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_val_loss = float('inf')
    best_macro_f1 = 0.0
    
    # Data paths
    train_chunk_folder = ['preprocessed_train', 'preprocessed_train_aug']
    val_chunk_folder = 'preprocessed_val'
    train_chunk_files = gather_chunks(train_chunk_folder)
    val_chunk_files = gather_chunks(val_chunk_folder)
    
    # Multiprocessing setup
    queue = mp.Queue(maxsize=3)
    processes = {}
    results = {}
    
    print(f'🧠 Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_chunk_files)
        epoch_loss = []
        
        # Preload chunks
        preload = min(3, len(train_chunk_files))
        for i in range(preload):
            wait_for_memory(threshold=96, interval=1)
            p = mp.Process(target=mp_async_load, args=(i, train_chunk_files[i], queue))
            p.start()
            processes[i] = p
        
        for chunk_idx, chunk_path in enumerate(train_chunk_files):
            # Collect queue results
            try:
                while chunk_idx not in results:
                    idx, status, payload = queue.get(timeout=300)
                    results[idx] = (status, payload)
            except Empty:
                print(f"⏰ Timeout waiting for chunk {chunk_idx}")
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
                print(f'❌ Failed to preload {chunk_path}: {payload}')
                continue
            
            lmdb_path = payload
            del payload
            
            # Create dataset and dataloader
            dataset = LMDBChunkDataset(lmdb_path, transform_tight=base_transforms, 
                                     transform_context=base_transforms)
            
            # Use original sampler setup
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
            
            # Use weighted sampler if enabled
            if config.get('use_weighted_sampler', True):
                weights, counts = build_sampler_weights(
                    lmdb_path,
                    dataset.seq_ids,
                    cross_pow=config['sampler_powers']["crosses"],
                    action_pow=config['sampler_powers']["actions"],
                    look_pow=config['sampler_powers']["looks"],
                )
                
                sampler = WeightedRandomSampler(
                    weights=torch.DoubleTensor(weights),
                    num_samples=len(weights),
                    replacement=True,
                )
                loader_kwargs["sampler"] = sampler
                loader_kwargs["shuffle"] = False
                
                print(f"📊 Sampler counts: actions={dict(counts['actions'])} "
                     f"looks={dict(counts['looks'])} crosses={dict(counts['crosses'])}")
            
            loader = DataLoader(dataset, **loader_kwargs)
            print(f"\n[Chunk {chunk_idx + 1}/{len(train_chunk_files)}] "
                 f"Training {len(loader)} batches from {chunk_path}")
            
            # Train chunk with enhanced function
            avg_loss = train_one_chunk_enhanced(
                model, loader, criterion, optimizer, device,
                loss_weight=config['loss_weight'],
                scaler=scaler,
                use_amp=use_amp,
                use_pin_memory=use_pin_memory,
                dynamic_weighting=dynamic_weighting,
                hard_negative_mining=hard_negative_mining
            )
            epoch_loss.append(avg_loss)
            
            # Cleanup
            del lmdb_path, dataset, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            trash = gc.collect()
            
            # Preload next chunk
            next_idx = chunk_idx + preload
            if next_idx < len(train_chunk_files):
                wait_for_memory(threshold=96, interval=1)
                p = mp.Process(target=mp_async_load, args=(next_idx, train_chunk_files[next_idx], queue))
                p.start()
                processes[next_idx] = p
            
            preload = min(preload, 3)
        
        # Cleanup remaining processes
        remaining = len(processes)
        for _ in range(remaining):
            try:
                idx, status, payload = queue.get(timeout=2)
                if status == 'ok':
                    del payload
                results.pop(idx, None)
            except Exception:
                pass
        
        for idx, proc in list(processes.items()):
            proc.join()
            processes.pop(idx, None)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Calculate epoch loss
        if len(epoch_loss) == 0:
            avg_epoch_loss = float('nan')
        else:
            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f'model_outputs/model_epoch{epoch+1}_{datetime_str}.pth')
        
        # Validation with enhanced metrics
        print("🔍 Running validation with enhanced metrics...")
        total_val_loss_sum = 0.0
        total_val_samples = 0
        all_val_predictions = {'actions': [], 'looks': [], 'crosses': []}
        all_val_targets = {'actions': [], 'looks': [], 'crosses': []}
        
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
                model, val_loader, criterion, device,
                use_amp=use_amp, use_pin_memory=use_pin_memory
            )
            
            total_val_loss_sum += chunk_loss_sum
            total_val_samples += chunk_n
            
            # Collect predictions for F1 calculation
            model.eval()
            with torch.inference_mode():
                for images_tight, images_context, motions, labels in val_loader:
                    images_tight = images_tight.to(device, non_blocking=use_pin_memory)
                    images_context = images_context.to(device, non_blocking=use_pin_memory)
                    motions = motions.to(device, non_blocking=use_pin_memory)
                    
                    outputs = model(images_tight, images_context, motions)
                    
                    for task in ['actions', 'looks', 'crosses']:
                        if task == 'crosses':
                            if model.cross_attention.use_frame_crosses:
                                logits = outputs['crosses_frame']
                            else:
                                logits = outputs['crosses_pooled']
                        else:
                            logits = outputs[task]
                        
                        _, preds = torch.max(logits, 1)
                        all_val_predictions[task].extend(preds.cpu().numpy())
                        all_val_targets[task].extend(labels[task].numpy())
            
            del val_dataset, val_loader, chunk_corrects
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        if total_val_samples == 0:
            raise RuntimeError("No validation samples found!")
        
        val_loss = total_val_loss_sum / total_val_samples
        
        # Compute enhanced metrics
        val_metrics = {}
        macro_f1_scores = []
        
        for task in ['actions', 'looks', 'crosses']:
            if len(set(all_val_targets[task])) > 1:
                val_metrics[f'{task}_f1'] = f1_score(all_val_targets[task], all_val_predictions[task], 
                                                   average='binary', pos_label=1)
                val_metrics[f'{task}_precision'] = precision_score(all_val_targets[task], all_val_predictions[task], 
                                                                 average='binary', pos_label=1)
                val_metrics[f'{task}_recall'] = recall_score(all_val_targets[task], all_val_predictions[task], 
                                                             average='binary', pos_label=1)
                val_metrics[f'{task}_accuracy'] = (np.array(all_val_predictions[task]) == 
                                                  np.array(all_val_targets[task])).mean()
                macro_f1_scores.append(val_metrics[f'{task}_f1'])
            else:
                val_metrics[f'{task}_f1'] = 0.0
                val_metrics[f'{task}_precision'] = 0.0
                val_metrics[f'{task}_recall'] = 0.0
                val_metrics[f'{task}_accuracy'] = (np.array(all_val_predictions[task]) == 
                                                  np.array(all_val_targets[task])).mean()
                macro_f1_scores.append(0.0)
        
        macro_f1 = np.mean(macro_f1_scores) if macro_f1_scores else 0.0
        overall_acc = np.mean([val_metrics[f'{task}_accuracy'] for task in ['actions', 'looks', 'crosses']])
        
        # Update dynamic weighting if enabled
        if dynamic_weighting is not None:
            task_metrics = {task: val_metrics[f'{task}_f1'] for task in ['actions', 'looks', 'crosses']}
            dynamic_weighting.update_performance(task_metrics)
        
        scheduler.step(val_loss)
        
        # Log enhanced metrics
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                epoch + 1, round(avg_epoch_loss, 4),
                round(val_metrics['actions_accuracy'], 4), round(val_metrics['actions_f1'], 4),
                round(val_metrics['actions_precision'], 4), round(val_metrics['actions_recall'], 4),
                round(val_metrics['looks_accuracy'], 4), round(val_metrics['looks_f1'], 4),
                round(val_metrics['looks_precision'], 4), round(val_metrics['looks_recall'], 4),
                round(val_metrics['crosses_accuracy'], 4), round(val_metrics['crosses_f1'], 4),
                round(val_metrics['crosses_precision'], 4), round(val_metrics['crosses_recall'], 4),
                round(val_loss, 4), round(overall_acc, 4), round(macro_f1, 4)
            ]
            writer.writerow(row)
        
        # Print metrics summary
        print(f"\n📊 Epoch {epoch + 1} Validation Results:")
        print(f"  Actions: Acc={val_metrics['actions_accuracy']:.3f}, F1={val_metrics['actions_f1']:.3f}")
        print(f"  Looks:   Acc={val_metrics['looks_accuracy']:.3f}, F1={val_metrics['looks_f1']:.3f}")
        print(f"  Crosses: Acc={val_metrics['crosses_accuracy']:.3f}, F1={val_metrics['crosses_f1']:.3f}")
        print(f"  Macro F1: {macro_f1:.3f}, Overall Acc: {overall_acc:.3f}")
        
        # Save best model based on macro F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            print(f"🏆 New best macro F1: {best_macro_f1:.4f}. Saving model...")
            torch.save(model.state_dict(), 
                      f'best_model_outputs/best_model_macro_f1_{macro_f1:.3f}_epoch{epoch+1}_{datetime_str}.pth')
        
        # Early stopping based on validation loss
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("⏹️ Early stopping triggered. Saving final model and stopping.")
            torch.save(model.state_dict(), 
                      f'model_outputs/final_model_epoch{epoch+1}_{datetime_str}.pth')
            break
        
        time.sleep(1)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()