import io, lmdb, pickle, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LMDBChunkDataset(Dataset):
    """
    Mirror of PTChunkDataset, but loads from LMDB chunk file.
    """
    def __init__(self, lmdb_path, transform_tight=None, transform_context=None):
        self.lmdb_path = lmdb_path
        self.env = None
        self.default_transform = transforms.ToTensor()
        self.transform_tight = transform_tight or self.default_transform
        self.transform_context = transform_context or self.default_transform

        # Build a list of available sequence IDs

        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            self.seq_ids = [key.decode().split("_")[0] for key, _ in cursor if key.decode().endswith("_meta")]
        env.close()
        print(f"[LMDBChunkDataset] Loaded index from {lmdb_path}: {len(self.seq_ids)} sequences")

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        self._ensure_env()
        seq_id = self.seq_ids[idx]
        with self.env.begin(write=False) as txn:
            meta_buf = txn.get(f"{seq_id}_meta".encode())
            if meta_buf is None:
                raise KeyError(f"Missing meta for seq {seq_id} in {self.lmdb_path}")
            meta = pickle.loads(meta_buf)
            motions = torch.as_tensor(meta.get("motions"))
            actions = torch.as_tensor(meta.get("actions"))
            looks = torch.as_tensor(meta.get("looks"))
            crosses = torch.as_tensor(meta.get("crosses"))

        # Determine frame count from motion tensor
            T = motions.shape[0]
            imgs_tight, imgs_context = [], []

            for k in range(T):
                tbuf = txn.get(f"{seq_id}_{k}_tight".encode())
                cbuf = txn.get(f"{seq_id}_{k}_context".encode())
                if tbuf is None or cbuf is None:
                    raise RuntimeError(f"Missing frame {k} for seq {seq_id} in {self.lmdb_path}")
                timg = Image.open(io.BytesIO(tbuf)).convert("RGB")
                cimg = Image.open(io.BytesIO(cbuf)).convert("RGB")
                imgs_tight.append(self.transform_tight(timg))
                imgs_context.append(self.transform_context(cimg))

        if len(imgs_tight) == 0:
            raise RuntimeError(f"Empty image sequence for seq {seq_id} in {self.lmdb_path}")
        
        return {
            "images_tight": torch.stack(imgs_tight),
            "images_context": torch.stack(imgs_context),
            "motions": motions,
            "actions": actions,
            "looks": looks,
            "crosses": crosses,
        }
    def __del__(self):
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
        except Exception:
            pass
