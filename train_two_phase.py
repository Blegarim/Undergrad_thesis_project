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
from queue import Empty

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
        images_tight = images_tight.to(device)
        images_context = images_context.to(device)
        motions = motions.to(device)
        labels = {k: v.to(device).long() for k, v in labels.items()}
        
        remap_cross_labels(labels)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images_tight, images_context, motions)
            loss = sum(criterion[t](outputs['crosses_frame'] if t == 'crosses' else outputs[t], labels[t])
                      for t in ['actions', 'looks', 'crosses'])
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    if len(dataloader) == 0:
        return float('nan')
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    all_preds = {k: [] for k in ['actions', 'looks', 'crosses']}
    all_targets = {k: [] for k in ['actions', 'looks', 'crosses']}
    
    with torch.inference_mode():
        for images_tight, images_context, motions, labels in dataloader:
            images_tight = images_tight.to(device)
            images_context = images_context.to(device)
            motions = motions.to(device)
            labels = {k: v.to(device).long() for k, v in labels.items()}
            
            remap_cross_labels(labels)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images_tight, images_context, motions)
            
            for t in ['actions', 'looks', 'crosses']:
                logits = outputs['crosses_frame'] if t == 'crosses' else outputs[t]
                preds = logits.argmax(dim=1)
                all_preds[t].extend(preds.cpu().numpy())
                all_targets[t].extend(labels[t].cpu().numpy())
    
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
        if 'classifier' not in name and 'crosses_frame_head' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}")
    
    datetime_str = datetime.now().strftime("%m%d_%H%M")
    log_file = f'training_log/training_log_two_phase_{datetime_str}.csv'
    os.makedirs('training_log', exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Phase', 'Epoch', 'Actions_F1', 'Looks_F1', 'Crosses_F1', 'Macro_F1'
        ])
    
    transforms_base = transforms.Compose([
        transforms.ToTensor(),
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
    num_workers = 4
    
    phase1_chunks = gather_chunks(['preprocessed_train_balanced'])
    phase2_chunks = gather_chunks(['preprocessed_train_augmented', 'preprocessed_train_augmented_dataaug'])
    val_chunks = gather_chunks('preprocessed_val')
    
    print(f"Phase 1 chunks: {len(phase1_chunks)}")
    print(f"Phase 2 chunks: {len(phase2_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    
    criterion = build_criterion(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_macro_f1 = 0
    best_model_path = None
    
    # ==================== PHASE 1: Balanced Training ====================
    print("\n" + "="*50)
    print("PHASE 1: Balanced Training")
    print("="*50)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(10):
        print(f"\nPhase 1 - Epoch {epoch+1}/10")
        random.shuffle(phase1_chunks)
        
        epoch_losses = []
        for chunk_path in tqdm(phase1_chunks, desc='Phase 1'):
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
            loss = train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=use_amp)
            epoch_losses.append(loss)
            del dataset, loader
            gc.collect()
            torch.cuda.empty_cache()
        
        val_metrics = []
        for chunk_path in val_chunks:
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
            val_metrics.append(validate(model, loader, criterion, device, use_amp=use_amp))
            del dataset, loader
            gc.collect()

        if not val_metrics:
            print(f"Warning: no validation metrics collected in Phase 1 epoch {epoch+1}, skipping.")
            continue

        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
        print(f"Loss: {np.mean(epoch_losses):.4f} | F1: A={avg_metrics['actions_f1']:.3f} L={avg_metrics['looks_f1']:.3f} C={avg_metrics['crosses_f1']:.3f} M={avg_metrics['macro_f1']:.3f}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow(['Phase 1', epoch+1, avg_metrics['actions_f1'], avg_metrics['looks_f1'],
                                   avg_metrics['crosses_f1'], avg_metrics['macro_f1']])

        if avg_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = avg_metrics['macro_f1']
            best_model_path = f'model_outputs/phase1_best_{datetime_str}.pth'
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(np.mean(epoch_losses))
        early_stopping(1 - avg_metrics['macro_f1'])
        if early_stopping.early_stop:
            print("Early stopping triggered in Phase 1.")
            break

    # ==================== PHASE 2: Full Data Fine-tuning ====================
    print("\n" + "="*50)
    print("PHASE 2: Full Data Fine-tuning")
    print("="*50)

    if best_model_path is None:
        raise RuntimeError("Phase 1 produced no improvement over 0 F1 — no checkpoint was saved.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(20):
        print(f"\nPhase 2 - Epoch {epoch+1}/20")
        random.shuffle(phase2_chunks)
        
        epoch_losses = []
        for chunk_path in tqdm(phase2_chunks, desc='Phase 2'):
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
            loss = train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=use_amp)
            epoch_losses.append(loss)
            del dataset, loader
            gc.collect()
            torch.cuda.empty_cache()
        
        val_metrics = []
        for chunk_path in val_chunks:
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
            val_metrics.append(validate(model, loader, criterion, device, use_amp=use_amp))
            del dataset, loader
            gc.collect()

        if not val_metrics:
            print(f"Warning: no validation metrics collected in Phase 2 epoch {epoch+1}, skipping.")
            continue

        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
        print(f"Loss: {np.mean(epoch_losses):.4f} | F1: A={avg_metrics['actions_f1']:.3f} L={avg_metrics['looks_f1']:.3f} C={avg_metrics['crosses_f1']:.3f} M={avg_metrics['macro_f1']:.3f}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow(['Phase 2', epoch+1, avg_metrics['actions_f1'], avg_metrics['looks_f1'],
                                   avg_metrics['crosses_f1'], avg_metrics['macro_f1']])

        if avg_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = avg_metrics['macro_f1']
            best_model_path = f'model_outputs/phase2_best_{datetime_str}.pth'
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(np.mean(epoch_losses))
        early_stopping(1 - avg_metrics['macro_f1'])
        if early_stopping.early_stop:
            print("Early stopping triggered in Phase 2.")
            break

    # ==================== PHASE 3: Decoupled Training ====================
    print("\n" + "="*50)
    print("PHASE 3: Decoupled Training")
    print("="*50)

    if best_model_path is None:
        raise RuntimeError("No checkpoint was saved through Phase 2 — cannot start Phase 3.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    freeze_backbone(model)
    
    classifier_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(classifier_params)} classifier parameters")
    
    optimizer = optim.Adam(classifier_params, lr=5e-5, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    for epoch in range(5):
        print(f"\nPhase 3 - Epoch {epoch+1}/5")
        random.shuffle(phase2_chunks)
        
        epoch_losses = []
        for chunk_path in tqdm(phase2_chunks, desc='Phase 3'):
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
            loss = train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=use_amp)
            epoch_losses.append(loss)
            del dataset, loader
            gc.collect()
            torch.cuda.empty_cache()
        
        val_metrics = []
        for chunk_path in val_chunks:
            dataset = LMDBChunkDataset(chunk_path, transforms_base, transforms_base)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
            val_metrics.append(validate(model, loader, criterion, device, use_amp=use_amp))
            del dataset, loader
            gc.collect()

        if not val_metrics:
            print(f"Warning: no validation metrics collected in Phase 3 epoch {epoch+1}, skipping.")
            continue

        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
        print(f"Loss: {np.mean(epoch_losses):.4f} | F1: A={avg_metrics['actions_f1']:.3f} L={avg_metrics['looks_f1']:.3f} C={avg_metrics['crosses_f1']:.3f} M={avg_metrics['macro_f1']:.3f}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow(['Phase 3', epoch+1, avg_metrics['actions_f1'], avg_metrics['looks_f1'],
                                   avg_metrics['crosses_f1'], avg_metrics['macro_f1']])

        if avg_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = avg_metrics['macro_f1']
            best_model_path = f'model_outputs/final_model_{datetime_str}.pth'
            torch.save(model.state_dict(), best_model_path)
        early_stopping(1 - avg_metrics['macro_f1'])
        if early_stopping.early_stop:
            print("Early stopping triggered in Phase 3.")
            break

    unfreeze_all(model)
    print(f"\nDone. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Log: {log_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
