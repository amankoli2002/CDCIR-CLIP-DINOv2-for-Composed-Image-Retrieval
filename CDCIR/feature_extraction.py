"""
Feature extraction for CLIP and DINOv2
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

TARGET_RATIO = 1.25
INPUT_DIM = 224


class TargetPad:
    """Pad image to square, scaled by target_ratio."""
    def __init__(self, ratio, dim):
        self.ratio = ratio

    def __call__(self, img):
        w, h = img.size
        mx = max(w, h)
        td = int(mx * self.ratio)
        pad = Image.new("RGB", (td, td), (255, 255, 255))
        pad.paste(img, ((td - w) // 2, (td - h) // 2))
        return pad


# Transforms
clip_transform = transforms.Compose([
    TargetPad(TARGET_RATIO, INPUT_DIM),
    transforms.Resize(INPUT_DIM, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(INPUT_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)),
])

dino_transform = transforms.Compose([
    TargetPad(TARGET_RATIO, INPUT_DIM),
    transforms.Resize(INPUT_DIM, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(INPUT_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)),
])


@torch.no_grad()
def encode_text_tokens(clip_model, tokens):
    """
    Extract all 77 token-level features from CLIP text encoder.
    Returns: [B, 77, clip_dim]
    """
    cast_dtype = clip_model.transformer.get_cast_dtype()
    x = clip_model.token_embedding(tokens).to(cast_dtype)
    x = x + clip_model.positional_embedding.to(cast_dtype)
    attn_mask = getattr(clip_model, 'attn_mask', None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device=x.device, dtype=x.dtype)
    x = clip_model.transformer(x, attn_mask=attn_mask)
    x = clip_model.ln_final(x)
    return x.float()


@torch.no_grad()
def extract_image_features(pil_img, clip_model, dinov2_model, device):
    """
    Extract all features needed from a single PIL image.

    Returns:
        clip_cls:     [512]       - CLIP CLS token, L2-normalized
        dino_patches: [256, 384]  - DINOv2 spatial patch tokens, fp16
                                    (fp16 matches the training cache format)
    """
    img = pil_img.convert("RGB")

    # CLIP CLS
    clip_input = clip_transform(img).unsqueeze(0).to(device)
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        clip_feat = clip_model.encode_image(clip_input)
    clip_cls = F.normalize(clip_feat.float(), dim=-1).cpu().squeeze(0)

    # DINOv2 patches
    dino_input = dino_transform(img).unsqueeze(0).to(device)
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        dino_out = dinov2_model.forward_features(dino_input)

    dino_patches = dino_out["x_norm_patchtokens"].float().cpu().half().squeeze(0)

    return clip_cls, dino_patches


@torch.no_grad()
def extract_text_features(text_str, clip_model, tokenizer, device):
    """
    Extract text features matching the training pipeline.

    Returns:
        text_tokens: [77, 512] - Full token-level features
        text_cls:    [512]     - CLIP text CLS (projected EOS), L2-normalized
    """
    tokens = tokenizer([text_str]).to(device)

    # Token-level features (77 × 512)
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        text_tok = encode_text_tokens(clip_model, tokens)
    text_tokens = text_tok.cpu().half().squeeze(0)  # [77, 512]

    # CLS feature (projected EOS)
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        text_feat = clip_model.encode_text(tokens)
    text_cls = F.normalize(text_feat.float(), dim=-1).cpu().squeeze(0)  # [512]

    return text_tokens, text_cls
