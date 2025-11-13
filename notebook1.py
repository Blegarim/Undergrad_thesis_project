import lmdb, pickle, io, torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from scripts.lmdb_dataset import LMDBChunkDataset

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config

#---- CONFIG ----
lmdb_path = "preprocessed_train_lmdb/chunk_001500.lmdb"
batch_size = 8
vit_args = vit_args_config()
motion_enc_args = motion_enc_args_config()
num_classes_dict = {
            'actions': 2,
            'looks': 2,
            'crosses': 2
        }
device = "cuda" if torch.cuda.is_available() else "cpu"

#---- DATASET TEST ----
print(f"Opening LMDB: {lmdb_path}")
dataset = LMDBChunkDataset(lmdb_path=lmdb_path,
                           transforms_tight=transforms.ToTensor(),
                           transforms_context=transforms.ToTensor())
print(f"Dataset length: {len(dataset)} sequences\n")

#---- Inspect ----
sample = dataset[0]
print("Single sample strucuture: ")
for k, v in sample.items():
    if torch.is_tensor(v):
        print(f"  {k:15s}  shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {k:15s}  type={type(v)}")

print("\nMotion tensor preview:\n", sample["motions"][:3])
print("Labels:", {k: int(v) for k,v in sample.items() if k in ["actions","looks","crosses"]})

#---- DATALOADER TEST ----
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
batch = next(iter(loader))

print("\nBatched shapes:")
print("images_tight :", batch["images_tight"].shape)
print("images_context:", batch["images_context"].shape)
print("motions       :", batch["motions"].shape)
print("actions       :", batch["actions"].shape, batch["actions"].unique())
print("looks         :", batch["looks"].shape, batch["looks"].unique())
print("crosses       :", batch["crosses"].shape, batch["crosses"].unique())

#---- MODEL TEST ----
model = EnsembleModel(
        motion_enc=MotionEncoder(**motion_enc_args),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(d_model=128, num_heads=4, num_classes_dict=num_classes_dict)
    ).to(device)
batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
with torch.no_grad():
    outputs = model(batch_gpu["images_tight"], batch_gpu["images_context"], batch_gpu["motions"])
for k,v in outputs.items():
    print(f"output[{k}] :", v.shape, v.dtype)

print("\n✅ Sanity check completed.")