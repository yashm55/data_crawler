import cv2
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def embed_video_streaming(video_path, model, processor, device, fps=2, batch_size=16):
    """
    Stream frames from a video file, embed in batches, and return a (N, D)
    embedding matrix — WITHOUT ever holding all frames in RAM at once.
    fps: frames per second to sample (use 0 for every frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval   = max(1, int(round(video_fps / fps))) if fps > 0 else 1
    n_expected = total // interval

    print(f"  Video FPS: {video_fps:.1f}  |  Sampling every {interval} frames  |  ~{n_expected} embeddings")

    all_embeddings = []
    batch, frame_idx, sampled = [], 0, 0

    with tqdm(total=n_expected, desc="  Embedding frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch.append(pil)
                sampled += 1

                if len(batch) == batch_size:
                    all_embeddings.append(_embed_batch(batch, model, processor, device))
                    batch.clear()
                    pbar.update(batch_size)

            frame_idx += 1
            del frame            # free OpenCV buffer immediately

        # flush remaining
        if batch:
            all_embeddings.append(_embed_batch(batch, model, processor, device))
            pbar.update(len(batch))

    cap.release()
    print(f"  Total frames embedded: {sampled}")
    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512))


def _embed_batch(pil_images, model, processor, device):
    """Embed a small batch of PIL images and return normalized numpy array."""
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
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
    return feats.cpu().numpy()


def load_folder_images(folder_path):
    """Load all images in a folder as PIL Images."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for p in sorted(Path(folder_path).iterdir()):
        if p.suffix.lower() in exts:
            images.append(Image.open(p).convert("RGB"))
    return images


def embed_images(images, model, processor, device, batch_size=32):
    """Return a (N, D) normalized embedding matrix for a list of PIL Images."""
    all_embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="  Embedding", leave=False):
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


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(sim_matrix):
    """Given a (M_video, N_query) matrix return a dict of analytics."""
    # Best match for each video frame → best query image
    best_per_frame = sim_matrix.max(axis=1)
    return {
        "avg_best":    float(best_per_frame.mean()),
        "max":         float(best_per_frame.max()),
        "min":         float(best_per_frame.min()),
        "median":      float(np.median(best_per_frame)),
        "std":         float(best_per_frame.std()),
        # fraction of video frames with ≥ 0.8 match
        "pct_high":    float((best_per_frame >= 0.8).mean() * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_query(video_path, query_images_root, output_dir, fps=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_model(device)

    # ── embed query video (streaming — no OOM) ──
    print(f"\nEmbedding query video (streaming @ {fps} FPS): {video_path}")
    video_emb = embed_video_streaming(video_path, model, processor, device, fps=fps)

    # ── iterate over query_images folders ──
    query_root = Path(query_images_root)
    folders = sorted([f for f in query_root.iterdir() if f.is_dir()])
    if not folders:
        print(f"No subfolders found in '{query_images_root}'.")
        return

    results = []
    for folder in folders:
        images = load_folder_images(folder)
        if not images:
            print(f"  [!] No images in {folder.name}, skipping.")
            continue

        print(f"\nComparing with [{folder.name}] ({len(images)} images)...")
        query_emb = embed_images(images, model, processor, device)   # (N, D)

        sim_matrix = cosine_similarity(video_emb, query_emb)        # (M, N)
        stats = compute_stats(sim_matrix)
        results.append({"name": folder.name, "stats": stats, "sim_matrix": sim_matrix})

    # ── print summary table ──
    print("\n" + "═" * 90)
    print(f"{'Query Folder':<40} {'Avg Best':>9} {'Max':>7} {'Median':>8} {'Std':>7} {'%≥0.8':>6}")
    print("─" * 90)
    results.sort(key=lambda r: r["stats"]["avg_best"], reverse=True)
    for r in results:
        s = r["stats"]
        print(
            f"{r['name']:<40} {s['avg_best']:>9.4f} {s['max']:>7.4f} "
            f"{s['median']:>8.4f} {s['std']:>7.4f} {s['pct_high']:>6.1f}%"
        )
    print("═" * 90)

    # ── bar chart ──
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    names      = [r["name"] for r in results]
    avg_scores = [r["stats"]["avg_best"] for r in results]
    max_scores = [r["stats"]["max"] for r in results]
    pct_high   = [r["stats"]["pct_high"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(names) * 0.8), 14))
    fig.suptitle(f"Video vs Query Images Similarity Report\n{Path(video_path).name}", fontsize=14, fontweight="bold")

    x = np.arange(len(names))
    width = 0.6

    # Plot 1: avg best score
    bars = axes[0].bar(x, avg_scores, width, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axhline(0.8, color="red", linestyle="--", linewidth=0.8, label="0.8 threshold")
    axes[0].set_ylabel("Avg Best Cosine Similarity")
    axes[0].set_title("Average Best Frame Similarity per Query Folder")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    for bar, val in zip(bars, avg_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Plot 2: max score
    bars2 = axes[1].bar(x, max_scores, width, color="darkorange", edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Max Cosine Similarity")
    axes[1].set_title("Peak Frame Similarity per Query Folder")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.05)
    for bar, val in zip(bars2, max_scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Plot 3: % high match frames
    bars3 = axes[2].bar(x, pct_high, width, color="seagreen", edgecolor="white", linewidth=0.5)
    axes[2].set_ylabel("% Video Frames with similarity ≥ 0.8")
    axes[2].set_title("% Matching Frames per Query Folder (threshold = 0.8)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylim(0, 105)
    for bar, val in zip(bars3, pct_high):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "similarity_report.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nChart saved to: {chart_path}")

    # ── top match heat-map for the best folder ──
    best = results[0]
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(best["sim_matrix"], aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label="Cosine Similarity")
    ax2.set_title(f"Similarity Matrix: Video vs best match [{best['name']}]")
    ax2.set_xlabel("Query Image Index")
    ax2.set_ylabel("Video Frame Index")
    heatmap_path = os.path.join(output_dir, "best_match_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a video against all folders in query_images and report similarity scores."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--query_images", default="query_images",
        help="Root folder containing per-video query image subfolders (default: query_images)"
    )
    parser.add_argument(
        "--output", default="reports",
        help="Directory to save charts and heatmaps (default: reports)"
    )
    parser.add_argument(
        "--fps", type=float, default=2,
        help="Frames per second to sample from the video (default: 2). Use 0 for every frame."
    )
    args = parser.parse_args()

    compare_with_query(args.video, args.query_images, args.output, fps=args.fps)
