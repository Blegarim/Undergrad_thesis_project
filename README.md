# Pedestrian Behavior Prediction

Multimodal deep learning project for predicting pedestrian actions (crossing/not crossing), looks, and crossing behaviors from video sequences.

## Architecture

- **Vision Transformer (ViT)**: Hierarchical ViT for visual feature extraction from pedestrian image sequences
- **Motion Encoder**: Temporal ConvNet-GRU for motion pattern encoding
- **Cross-Attention Module**: Multimodal fusion of visual and motion features
- **Ensemble Model**: Final prediction head with multi-task outputs

## Quick Start

### Training
```bash
# Standard training
python train.py

# Enhanced training with class imbalance handling
python train_enhanced.py --preset recommended --enable_advanced True
```

### Testing
```bash
python test.py
python test.py --model_type motion_only  # Ablation study variants
python test.py --model_type visual_only
python test.py --model_type vanilla_concat
```

### Inference
```bash
python main.py
```

## Project Structure

```
├── models/              # Model architecture definitions
│   ├── Vision_Transformer.py
│   ├── Motion_Encoder.py
│   ├── Cross_Attention_Module.py
│   ├── Unified_Module.py
│   └── AblationModels.py
├── scripts/             # Data processing utilities
│   ├── lmdb_dataset.py
│   ├── preprocess_data_lmdb.py
│   ├── pedestrian_detection.py
│   └── ...
├── config.py            # Configuration parameters
├── train.py             # Main training script
├── train_enhanced.py    # Enhanced training with class imbalance handling
├── test.py              # Evaluation script
└── main.py              # Video inference
```

## Requirements

```bash
pip install -r requirements.txt
```

See AGENTS.md for detailed development guidelines.
