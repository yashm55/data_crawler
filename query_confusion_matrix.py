import torch
import numpy as np
import argparse
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_id})...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def load_folder_images(folder_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [Image.open(p).convert("RGB") for p in sorted(Path(folder_path).iterdir()) if p.suffix.lower() in exts]


def embed_images(images, model, processor, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            feats = outputs
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            feats = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs[0] if isinstance(outputs, tuple) else outputs

        if hasattr(model, "visual_projection") and hasattr(model.config, "projection_dim"):
            if feats.shape[-1] != model.config.projection_dim:
                feats = model.visual_projection(feats)

        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512))


def folder_mean_embedding(folder, model, processor, device):
    """Return the mean embedding vector for all images in a folder."""
    images = load_folder_images(folder)
    if not images:
        return None
    emb = embed_images(images, model, processor, device)
    return emb.mean(axis=0, keepdims=True)  # (1, D)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(query_images_root, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_model(device)

    root = Path(query_images_root)
    folders = sorted([f for f in root.iterdir() if f.is_dir()])
    if not folders:
        print(f"No subfolders found in '{query_images_root}'.")
        return

    print(f"\nFound {len(folders)} folders. Embedding each...\n")
    names, mean_embs = [], []
    for folder in tqdm(folders, desc="Folders"):
        emb = folder_mean_embedding(folder, model, processor, device)
        if emb is None:
            print(f"  [!] Skipping empty folder: {folder.name}")
            continue
        names.append(folder.name)
        mean_embs.append(emb)

    # Stack into (N, D) and compute N×N pairwise cosine similarity
    mat = np.vstack(mean_embs)          # (N, D)
    sim = cosine_similarity(mat, mat)   # (N, N)

    N = len(names)

    # ── print table ──
    print("\n" + "─" * 60)
    print(f"Pairwise Similarity (diagonal = self, should be 1.0)")
    print("─" * 60)
    col_w = max(len(n) for n in names) + 2
    header = f"{'':>{col_w}}" + "".join(f"{n[:8]:>10}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"{name:>{col_w}}" + "".join(f"{sim[i, j]:>10.4f}" for j in range(N))
        print(row)

    # Summary stats (off-diagonal)
    off_diag = sim[~np.eye(N, dtype=bool)]
    print(f"\nOff-diagonal stats:")
    print(f"  Mean:   {off_diag.mean():.4f}")
    print(f"  Max:    {off_diag.max():.4f}")
    print(f"  Min:    {off_diag.min():.4f}")
    print(f"  Std:    {off_diag.std():.4f}")

    # ── confusion-matrix heat-map ──
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig_size = max(10, N * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(sim, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Cosine Similarity", fraction=0.046, pad=0.04)

    # Annotate each cell
    for i in range(N):
        for j in range(N):
            color = "black" if 0.55 < sim[i, j] < 0.90 else "white"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    short_names = [n[:18] for n in names]
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title("Query Folders — Pairwise Cosine Similarity Confusion Matrix\n(using mean frame embedding per folder)", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    out_path = Path(output_dir) / "query_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {out_path}")

    # ── top-N most similar pairs ──
    print("\nTop 10 most similar folder pairs (excluding self):")
    flat = [(sim[i, j], names[i], names[j]) for i in range(N) for j in range(i + 1, N)]
    flat.sort(reverse=True)
    for score, a, b in flat[:10]:
        print(f"  {score:.4f}  {a}  ↔  {b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a pairwise cosine similarity confusion matrix across all query image folders."
    )
    parser.add_argument(
        "--query_images", default="query_images",
        help="Root folder containing per-video query image subfolders (default: query_images)"
    )
    parser.add_argument(
        "--output", default="reports",
        help="Directory to save the confusion matrix image (default: reports)"
    )
    args = parser.parse_args()
    build_confusion_matrix(args.query_images, args.output)
