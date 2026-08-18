"""
Gallery management and retrieval logic.

The gallery is built from precomputed CLIP CLS features of all val images,
directly L2-normalized.

Retrieval = cosine similarity between composed query and gallery features.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


class Gallery:
    """Manages the precomputed gallery for fast retrieval."""

    def __init__(self, clip_cls_cache, image_dir, device):
        """
        Args:
            clip_cls_cache: dict {img_id: tensor [512]}
            image_dir: Path to val_images/ folder
            device: torch device
        """
        self.image_dir = Path(image_dir)
        self.device = device

        self.ids = list(clip_cls_cache.keys())

        # L2-normalize CLIP CLS features
        feats_list = []
        with torch.no_grad():
            for s in range(0, len(self.ids), 512):
                batch_ids = self.ids[s:s + 512]
                gc = torch.stack([clip_cls_cache[i] for i in batch_ids])
                feats_list.append(F.normalize(gc.float(), dim=-1))

        self.features = torch.cat(feats_list, 0)   # already normalized
        self.id2idx = {gid: i for i, gid in enumerate(self.ids)}

        # Build image path lookup 
        self._path_index = {}
        for p in self.image_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg"):
                self._path_index[p.stem] = p

        print(f"Gallery ready: {len(self.ids)} images, "
              f"{len(self._path_index)} image files found")

    def find_image(self, img_id):
        """Locate an image file by ID (searches subfolders via cached index)."""
        return self._path_index.get(img_id, None)

    @torch.no_grad()
    def retrieve(self, query_feat, top_k=10):
        """
        Retrieve top-K images given a composed query feature.

        Args:
            query_feat: tensor [1, dim] or [dim], L2-normalized
            top_k: number of results

        Returns:
            results: list of (img_id, similarity_score, PIL.Image or None)
        """
        if query_feat.dim() == 1:
            query_feat = query_feat.unsqueeze(0)
        query_feat = F.normalize(query_feat, dim=-1)

        sims = (query_feat @ self.features.T).squeeze(0)
        top_k = min(top_k, len(self.ids))
        topk_vals, topk_idxs = sims.topk(top_k)

        results = []
        for score, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            img_id = self.ids[idx]
            img_path = self.find_image(img_id)
            pil_img = None
            if img_path and img_path.exists():
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                except Exception:
                    pass
            results.append((img_id, score, pil_img))

        
        return results