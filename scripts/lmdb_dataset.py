import os
import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, ImageReadMode

class LMDBChunkDataset(Dataset):
    """
    Mirror of PTChunkDataset, but loads from LMDB chunk file.
    """
    def __init__(self, lmdb_path, transform_tight=None, transform_context=None):
        self.lmdb_path = lmdb_path
        self._env = None
        self._pid = None
        self.default_transform = torch.nn.Identity()
        self.transform_tight = transform_tight or self.default_transform
        self.transform_context = transform_context or self.default_transform

        # Build a list of available sequence IDs
        self.seq_ids = []
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        try:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode()
                    if key_str.endswith("_meta"):
                        seq_id = key_str[:-len("_meta")]
                        self.seq_ids.append(seq_id)
        finally:
            env.close()
        print(f"[LMDBChunkDataset] Loaded index from {lmdb_path}: {len(self.seq_ids)} sequences")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        if self._env is not None:
            self._env.close()

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            if self._env is not None:
                self._env.close()
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
            self._pid = pid
        return self._env

    def __len__(self):
        return len(self.seq_ids)

    @staticmethod
    def _decode_jpeg_buf(buf):
        """Decode JPEG bytes to float32 [C,H,W] tensor in [0,1] via libjpeg-turbo."""
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        return decode_jpeg(t, mode=ImageReadMode.RGB).float().div_(255.0)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        env = self._get_env()
        with env.begin(write=False) as txn:
            meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
            motions = meta["motions"]
            actions = meta["actions"]
            looks = meta["looks"]
            crosses = meta["crosses"]

            T = motions.shape[0]
            imgs_tight, imgs_context = [], []

            for k in range(T):
                tbuf = txn.get(f"{seq_id}_{k}_tight".encode())
                cbuf = txn.get(f"{seq_id}_{k}_context".encode())
                if tbuf is None or cbuf is None:
                    continue
                timg = self._decode_jpeg_buf(tbuf)
                cimg = self._decode_jpeg_buf(cbuf)
                if self.transform_tight:
                    timg = self.transform_tight(timg)
                if self.transform_context:
                    cimg = self.transform_context(cimg)
                imgs_tight.append(timg)
                imgs_context.append(cimg)

            if len(imgs_tight) != T:
                raise ValueError(
                    f"[LMDBChunkDataset] Sequence {seq_id!r}: expected {T} frames, "
                    f"found {len(imgs_tight)} — missing LMDB frame keys. "
                    f"Chunk may be corrupted: {self.lmdb_path}"
                )

        return {
            "images_tight": torch.stack(imgs_tight),
            "images_context": torch.stack(imgs_context),
            "motions": motions,
            "actions": actions,
            "looks": looks,
            "crosses": crosses,
        }
