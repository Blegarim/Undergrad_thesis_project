import io, lmdb, pickle, torch
from torch.utils.data import Dataset
from PIL import Image

class LMDBChunkDataset(Dataset):
    """
    Mirror of PTChunkDataset, but loads from LMDB chunk file.
    """
    def __init__(self, lmdb_path, transform_tight=None, transform_context=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.transform_tight = transform_tight
        self.transform_context = transform_context

        # Build a list of available sequence IDs
        self.seq_ids = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key_str = key.decode()
                if key_str.endswith("_meta"):
                    seq_id = key_str.split("_")[0]
                    self.seq_ids.append(seq_id)
        print(f"[LMDBChunkDataset] Loaded index from {lmdb_path}: {len(self.seq_ids)} sequences")

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        meta = pickle.loads(self.txn.get(f"{seq_id}_meta".encode()))
        motions = meta["motions"]
        actions = meta["actions"]
        looks = meta["looks"]
        crosses = meta["crosses"]

        # Determine frame count from motion tensor
        T = motions.shape[0]
        imgs_tight, imgs_context = [], []

        for k in range(T):
            tbuf = self.txn.get(f"{seq_id}_{k}_tight".encode())
            cbuf = self.txn.get(f"{seq_id}_{k}_context".encode())
            if tbuf is None or cbuf is None:
                continue
            timg = Image.open(io.BytesIO(tbuf)).convert("RGB")
            cimg = Image.open(io.BytesIO(cbuf)).convert("RGB")
            if self.transform_tight:
                timg = self.transform_tight(timg)
            if self.transform_context:
                cimg = self.transform_context(cimg)
            imgs_tight.append(timg)
            imgs_context.append(cimg)

        return {
            "images_tight": torch.stack(imgs_tight),
            "images_context": torch.stack(imgs_context),
            "motions": motions,
            "actions": actions,
            "looks": looks,
            "crosses": crosses,
        }
