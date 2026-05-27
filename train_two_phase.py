"""
Two-Phase Training with Decoupled Classification

Phase 1: Train on balanced subset - learn robust features
Phase 2: Fine-tune on full augmented data - refine decision boundary
Phase 3: Decouple - freeze backbone, train classifiers on full data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
from datetime import datetime
import gc
import random
import csv
import multiprocessing as mp
from tqdm import tqdm

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model
from scripts.lmdb_dataset import LMDBChunkDataset
from class_imbalance_strategies import FocalLoss
from scripts.train_utils import collate_fn, remap_cross_labels, gather_chunks, EarlyStopping


def build_criterion(device):
    return {
        'actions': FocalLoss(gamma=2.0).to(device),
        'looks': FocalLoss(alpha=[1.0, 5.0], gamma=2.0).to(device),
        'crosses': FocalLoss(alpha=[1.0, 20.0], gamma=2.5).to(device),
    }


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp=True):
    model.train()
    total_loss = 0

    for images_tight, images_context, motions, labels in dataloader:
        images_tight = images_tight.to(device, non_blocking=True)
        images_context = images_context.to(device, non_blocking=True)
        motions = motions.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True).long() for k, v in labels.items()}

        remap_cross_labels(labels)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images_tight, images_context, motions)
            loss = sum(criterion[t](outputs['crosses_frame'] if t == 'crosses' else outputs[t], labels[t])
                      for t in ['actions', 'looks', 'crosses'])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    if len(dataloader) == 0:
        return float('nan')
    return total_loss / len(dataloader)


def validate_all_chunks(model, val_chunks, criterion, device, batch_size, num_workers,
                        transform_tight, transform_context, use_amp=True):
    """Validate across all chunks, accumulating raw predictions globally before computing metrics."""
    model.eval()
    all_preds = {k: [] for k in ['actions', 'looks', 'crosses']}
    all_targets = {k: [] for k in ['actions', 'looks', 'crosses']}

    with torch.inference_mode():
        for chunk_path in val_chunks:
            dataset = LMDBChunkDataset(chunk_path, transform_tight, transform_context)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=(device.type == 'cuda'),
                              prefetch_factor=2 if num_workers > 0 else None)

            for images_tight, images_context, motions, labels in loader:
                images_tight = images_tight.to(device, non_blocking=True)
                images_context = images_context.to(device, non_blocking=True)
                motions = motions.to(device, non_blocking=True)
                labels = {k: v.to(device, non_blocking=True).long() for k, v in labels.items()}

                remap_cross_labels(labels)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images_tight, images_context, motions)

                for t in ['actions', 'looks', 'crosses']:
                    logits = outputs['crosses_frame'] if t == 'crosses' else outputs[t]
                    preds = logits.argmax(dim=1)
                    all_preds[t].extend(preds.cpu().numpy())
                    all_targets[t].extend(labels[t].cpu().numpy())

            del dataset, loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    metrics = {}
    for t in ['actions', 'looks', 'crosses']:
        if len(set(all_targets[t])) > 1:
            metrics[f'{t}_f1'] = f1_score(all_targets[t], all_preds[t], average='binary')
        else:
            metrics[f'{t}_f1'] = 0.0

    metrics['macro_f1'] = np.mean([metrics[f'{t}_f1'] for t in ['actions', 'looks', 'crosses']])
    return metrics


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'crosses_frame_head' not in name and 'pool_mlp' not in name:
            param.requires_grad = False


def run_phase(phase_name, model, chunks, val_chunks, criterion, optimizer, scheduler,
              scaler, early_stopping, device, batch_size, num_workers,
              transform_tight, transform_context, use_amp, max_epochs,
              best_macro_f1, save_prefix, datetime_str, log_file):
    """Run one training phase: train on chunks, validate, log, and save best model."""
    best_model_path = None

    for epoch in range(max_epochs):
        print(f"\n{phase_name} - Epoch {epoch+1}/{max_epochs}")
        random.shuffle(chunks)

        epoch_losses = []
        for chunk_path in tqdm(chunks, desc=phase_name):
            dataset = LMDBChunkDataset(chunk_path, transform_tight, transform_context)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=(device.type == 'cuda'))
            loss = train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=use_amp)
            epoch_losses.append(loss)
            del dataset, loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = np.mean(epoch_losses)
        metrics = validate_all_chunks(model, val_chunks, criterion, device,
                                      batch_size, num_workers,
                                      transform_tight, transform_context, use_amp)
        print(f"Loss: {avg_loss:.4f} | F1: A={metrics['actions_f1']:.3f} "
              f"L={metrics['looks_f1']:.3f} C={metrics['crosses_f1']:.3f} "
              f"M={metrics['macro_f1']:.3f}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                phase_name, epoch+1, round(avg_loss, 4),
                metrics['actions_f1'], metrics['looks_f1'],
                metrics['crosses_f1'], metrics['macro_f1']
            ])

        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']
            best_model_path = f'model_outputs/{save_prefix}_{datetime_str}.pth'
            torch.save(model.state_dict(), best_model_path)

        if scheduler is not None:
            scheduler.step(avg_loss)
        early_stopping(1 - metrics['macro_f1'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered in {phase_name}.")
            break

    return best_macro_f1, best_model_path


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}")

    datetime_str = datetime.now().strftime("%m%d_%H%M")
    log_file = f'training_log/training_log_two_phase_{datetime_str}.csv'
    os.makedirs('training_log', exist_ok=True)

    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Phase', 'Epoch', 'Train_Loss', 'Actions_F1', 'Looks_F1', 'Crosses_F1', 'Macro_F1'
        ])

    transform_tight = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_context = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    d_model = get_unified_dim_model()
    num_classes = {'actions': 2, 'looks': 2, 'crosses': 2}

    motion_enc = MotionEncoder(**motion_enc_args_config())
    vit = ViT_Hierarchical(**vit_args_config())
    cross_attention = CrossAttentionModule(
        d_model=d_model, num_heads=4, num_classes_dict=num_classes,
        use_frame_crosses=True, frame_pool='logsumexp'
    )
    model = EnsembleModel(motion_enc, vit, cross_attention, d_model).to(device)

    batch_size = 4
    num_workers = 6

    phase1_chunks = gather_chunks(['preprocessed_train_balanced'])
    phase2_chunks = gather_chunks(['preprocessed_train_augmented', 'preprocessed_train_augmented_dataaug'])
    val_chunks = gather_chunks('preprocessed_val')

    print(f"Phase 1 chunks: {len(phase1_chunks)}")
    print(f"Phase 2 chunks: {len(phase2_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")

    criterion = build_criterion(device)

    best_macro_f1 = 0
    os.makedirs('model_outputs', exist_ok=True)
    best_model_path = f'model_outputs/phase1_baseline_{datetime_str}.pth'
    torch.save(model.state_dict(), best_model_path)

    # ==================== PHASE 1: Balanced Training ====================
    print("\n" + "="*50)
    print("PHASE 1: Balanced Training")
    print("="*50)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    best_macro_f1, path = run_phase(
        'Phase 1', model, phase1_chunks, val_chunks, criterion, optimizer,
        scheduler, scaler, early_stopping, device, batch_size, num_workers,
        transform_tight, transform_context, use_amp, max_epochs=10,
        best_macro_f1=best_macro_f1, save_prefix='phase1_best',
        datetime_str=datetime_str, log_file=log_file,
    )
    if path is not None:
        best_model_path = path

    # ==================== PHASE 2: Full Data Fine-tuning ====================
    print("\n" + "="*50)
    print("PHASE 2: Full Data Fine-tuning")
    print("="*50)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    best_macro_f1, path = run_phase(
        'Phase 2', model, phase2_chunks, val_chunks, criterion, optimizer,
        scheduler, scaler, early_stopping, device, batch_size, num_workers,
        transform_tight, transform_context, use_amp, max_epochs=20,
        best_macro_f1=best_macro_f1, save_prefix='phase2_best',
        datetime_str=datetime_str, log_file=log_file,
    )
    if path is not None:
        best_model_path = path

    # ==================== PHASE 3: Decoupled Training ====================
    print("\n" + "="*50)
    print("PHASE 3: Decoupled Training")
    print("="*50)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    freeze_backbone(model)

    classifier_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(classifier_params)} classifier parameters")

    optimizer = optim.Adam(classifier_params, lr=5e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    best_macro_f1, path = run_phase(
        'Phase 3', model, phase2_chunks, val_chunks, criterion, optimizer,
        scheduler, scaler, early_stopping, device, batch_size, num_workers,
        transform_tight, transform_context, use_amp, max_epochs=5,
        best_macro_f1=best_macro_f1, save_prefix='final_model',
        datetime_str=datetime_str, log_file=log_file,
    )

    print(f"\nDone. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Log: {log_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
