
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from models import TextQueryDINOCombiner
from feature_extraction import extract_image_features, extract_text_features
from retrieval import Gallery

# Configuration 
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals 
clip_model = None
dinov2_model = None
tokenizer = None
combiner = None
gallery = None


def initialize():
    """Load all models and build gallery. Called once at startup."""
    global clip_model, dinov2_model, tokenizer
    global combiner, gallery

    import open_clip

    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")

    # 1. Load eval bundle 
    
    bundle = torch.load(DATA_DIR / "eval_bundle.pt",
                        map_location="cpu", weights_only=False)
    config = bundle["config"]
    
    print(f"  Dataset: {config['DATASET']} | "
          f"Epoch: {bundle['epoch']} | R@10: {bundle['r10']:.2f}%")

    # 2. Load CLIP 
   
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config["CLIP_MODEL"], pretrained=config["CLIP_PRETRAIN"])
    clip_model = clip_model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer(config["CLIP_MODEL"])

    #  3. Load DINOv2 
   
    os.makedirs("/tmp/torch_hub", exist_ok=True)
    torch.hub.set_dir("/tmp/torch_hub")
    dinov2_model = torch.hub.load(
        "facebookresearch/dinov2", config["DINO_MODEL"], trust_repo=True)
    dinov2_model = dinov2_model.to(DEVICE).eval()

    # Build combiner 
    combiner = TextQueryDINOCombiner(
        clip_dim=config["CLIP_DIM"],
        dino_dim=config["DINO_DIM"],
        num_heads=config["NUM_HEADS"],
        num_layers=config["NUM_LAYERS"],
        ffn_mult=config["FFN_MULT"],
        dropout=config["DROPOUT"],
        grid_size=config["GRID_SIZE"],
    ).to(DEVICE)
    combiner.load_state_dict(bundle["combiner"])
    combiner.eval()

    

    params = sum(p.numel() for p in combiner.parameters()) / 1e6
    print(f"  Combiner: {params:.2f}M params")

    # Build gallery
   
    clip_cls_cache = torch.load(DATA_DIR / "val_clip_cls.pt",
                                map_location="cpu", weights_only=False)

    image_dir = DATA_DIR / "val_images"
    if not image_dir.exists():
        print(f"WARNING: {image_dir} not found.")
        image_dir.mkdir(parents=True, exist_ok=True)

    gallery = Gallery(clip_cls_cache, image_dir, DEVICE)

    if DEVICE.type == "cuda":
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\nReady.\n")


@torch.no_grad()
def do_retrieval(input_image, text_query, top_k):
    """Core retrieval function called by the GUI."""
    if input_image is None:
        return []
    if not text_query or not text_query.strip():
        return []

    top_k = int(top_k)
    text_query = text_query.strip()

    # 1. Extract image features
    clip_cls, dino_patches = extract_image_features(
        input_image, clip_model, dinov2_model, DEVICE)

    # 2. Extract text features
    text_tokens, text_cls = extract_text_features(
        text_query, clip_model, tokenizer, DEVICE)

    # 3. Compose query
    txt = text_tokens.float().unsqueeze(0).to(DEVICE)
    pat = dino_patches.float().unsqueeze(0).to(DEVICE)
    rc = clip_cls.unsqueeze(0).to(DEVICE)
    tcls = text_cls.unsqueeze(0).to(DEVICE)

    composed, _ = combiner(txt, pat, rc, tcls)
    query_feat = F.normalize(composed, dim=-1).cpu()

    # 4. Retrieve
    results = gallery.retrieve(query_feat, top_k=top_k)

    # 5. Format for Gradio
    gallery_items = []
    for rank, (img_id, score, pil_img) in enumerate(results, 1):
        caption = f"#{rank}"
        if pil_img is not None:
            gallery_items.append((pil_img, caption))
        else:
            placeholder = Image.new("RGB", (224, 224), (40, 40, 40))
            gallery_items.append((placeholder, f"#{rank} {img_id[:20]}"))

    return gallery_items


# UI 
css = """
.main-title { text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #888; font-size: 14px;
            margin-top: 0; margin-bottom: 16px; }
"""

with gr.Blocks(css=css, title="Composed Image Retrieval") as demo:
    gr.HTML("<h2 class='main-title'>CDCIR</h2>")
    gr.HTML("<p class='subtitle'>Upload a reference image, describe how to "
            "modify it, and retrieve matching images from the gallery.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", label="Reference Image", height=280)
            text_input = gr.Textbox(
                label="Text Modification",
                placeholder="e.g. 'make it red and shorter'",
                lines=2)
            top_k_slider = gr.Slider(
                minimum=1, maximum=50, value=10, step=1,
                label="Number of results")
            search_btn = gr.Button("Search", variant="primary")

        with gr.Column(scale=2):
            result_gallery = gr.Gallery(
                label="Retrieved Images",
                columns=5, rows=2,
                object_fit="contain", height=520)

    search_btn.click(
        fn=do_retrieval,
        inputs=[input_image, text_input, top_k_slider],
        outputs=result_gallery)

    text_input.submit(
        fn=do_retrieval,
        inputs=[input_image, text_input, top_k_slider],
        outputs=result_gallery)


#  Launch 
initialize()
demo.launch()