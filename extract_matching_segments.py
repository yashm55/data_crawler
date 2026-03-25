"""
extract_matching_segments.py
─────────────────────────────
Scores every sampled frame of a video against query images using CLIP,
identifies contiguous sequences above a similarity threshold, and saves
each matched segment as a cropped video clip.

Usage:
    python extract_matching_segments.py video.mp4 --query_images query_images
    python extract_matching_segments.py video.mp4 --threshold 0.80 --fps 2 --min_seconds 1.5
"""

import cv2
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_id})...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def _embed_batch(pil_images, model, processor, device):
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


def embed_folder_images(folders, model, processor, device, batch_size=32):
    """Load and embed all images from a list of folders, return matrix + folder map."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []
    folder_map = [] # folder_map[i] = "folder_name" for i-th embedding
    
    for folder in folders:
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in exts:
                all_images.append(Image.open(p).convert("RGB"))
                folder_map.append(folder.name)

    if not all_images:
        return None, None
    
    print(f"  Embedding {len(all_images)} query images from {len(folders)} folder(s)...")
    all_embs = []
    for i in range(0, len(all_images), batch_size):
        all_embs.append(_embed_batch(all_images[i:i+batch_size], model, processor, device))
    
    emb_matrix = np.vstack(all_embs)
    return emb_matrix, folder_map


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame scoring (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def score_video_frames(video_path, query_emb_matrix, folder_map, model, processor, device,
                       fps=2, batch_size=16):
    """
    Stream video, compute per-frame similarity, return scores + metadata.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval  = max(1, int(round(video_fps / fps))) if fps > 0 else 1
    n_expected = total // interval
    print(f"  Video: {total} frames @ {video_fps:.1f} FPS  |  sampling every {interval} frames (~{n_expected} samples)")

    frame_scores = []
    batch_pil, batch_meta = [], []
    frame_idx = 0

    with tqdm(total=n_expected, desc="  Scoring frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_pil.append(pil)
                batch_meta.append({"frame_idx": frame_idx, "time_sec": frame_idx / video_fps})
                
                if len(batch_pil) == batch_size:
                    embs = _embed_batch(batch_pil, model, processor, device)
                    sims = cosine_similarity(embs, query_emb_matrix) # (B, N_query)
                    
                    scores = sims.max(axis=1)
                    best_indices = sims.argmax(axis=1)
                    
                    for meta, score, best_idx in zip(batch_meta, scores, best_indices):
                        frame_scores.append({
                            **meta, 
                            "score": float(score), 
                            "best_folder": folder_map[best_idx]
                        })
                    batch_pil.clear(); batch_meta.clear()
                    pbar.update(batch_size)
            frame_idx += 1
            del frame

        if batch_pil:
            embs = _embed_batch(batch_pil, model, processor, device)
            sims = cosine_similarity(embs, query_emb_matrix)
            scores = sims.max(axis=1)
            best_indices = sims.argmax(axis=1)
            for meta, score, best_idx in zip(batch_meta, scores, best_indices):
                frame_scores.append({
                    **meta, 
                    "score": float(score), 
                    "best_folder": folder_map[best_idx]
                })
            pbar.update(len(batch_pil))

    cap.release()
    return frame_scores, video_fps


# ─────────────────────────────────────────────────────────────────────────────
# Segment detection
# ─────────────────────────────────────────────────────────────────────────────

def find_segments(frame_scores, threshold, min_frames, gap_tolerance=2):
    """
    Find contiguous runs of frames with score >= threshold.

    gap_tolerance: allow this many consecutive below-threshold frames inside a
                   segment (handles brief cuts / logo flashes).
    min_frames: minimum number of qualifying frames a segment must have.
    Returns list of {start_frame, end_frame, start_sec, end_sec, avg_score, max_score, n_frames}
    """
    segments = []
    current   = None
    below_run = 0

    for fs in frame_scores:
        above = fs["score"] >= threshold

        if above:
            below_run = 0
            if current is None:
                current = {
                    "start_frame": fs["frame_idx"],
                    "start_sec":   fs["time_sec"],
                    "scores":      [],
                }
            current["scores"].append(fs["score"])
            current["end_frame"] = fs["frame_idx"]
            current["end_sec"]   = fs["time_sec"]
            current["best_folder"] = fs["best_folder"] # overwrite with latest for segment label

        else:
            if current is not None:
                below_run += 1
                if below_run > gap_tolerance:
                    # close segment
                    if len(current["scores"]) >= min_frames:
                        segments.append({
                            "start_frame": current["start_frame"],
                            "end_frame":   current["end_frame"],
                            "start_sec":   current["start_sec"],
                            "end_sec":     current["end_sec"],
                            "avg_score":   float(np.mean(current["scores"])),
                            "max_score":   float(np.max(current["scores"])),
                            "n_frames":    len(current["scores"]),
                            "best_folder": current["best_folder"]
                        })
                    current   = None
                    below_run = 0


    # close any open segment at end of video
    if current is not None and len(current["scores"]) >= min_frames:
        segments.append({
            "start_frame": current["start_frame"],
            "end_frame":   current["end_frame"],
            "start_sec":   current["start_sec"],
            "end_sec":     current["end_sec"],
            "avg_score":   float(np.mean(current["scores"])),
            "max_score":   float(np.max(current["scores"])),
            "n_frames":    len(current["scores"]),
            "best_folder": current["best_folder"]
        })

    return segments


def save_timeline_plot(frame_scores, segments, threshold, out_path):
    """Plot similarity vs time and highlight segments."""
    times = [fs["time_sec"] for fs in frame_scores]
    scores = [fs["score"] for fs in frame_scores]
    
    plt.figure(figsize=(15, 6))
    plt.plot(times, scores, label="Similarity Score", color="steelblue", alpha=0.8, linewidth=1.5)
    plt.axhline(y=threshold, color="red", linestyle="--", alpha=0.6, label=f"Threshold ({threshold})")
    
    # Highlight segments
    for seg in segments:
        plt.axvspan(seg["start_sec"], seg["end_sec"], color="green", alpha=0.2, label="Matched Segment" if seg == segments[0] else "")
        plt.text((seg["start_sec"] + seg["end_sec"])/2, max(scores)*1.02, seg["best_folder"], 
                 ha="center", fontsize=8, color="darkgreen", rotation=45)

    plt.xlabel("Time (seconds)")
    plt.ylabel("CLIP Cosine Similarity")
    plt.title("Search Match Timeline")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.ylim(min(scores)*0.9, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Timeline plot saved: {out_path}")



# ─────────────────────────────────────────────────────────────────────────────
# Video cropping
# ─────────────────────────────────────────────────────────────────────────────

def crop_segment(video_path, start_sec, end_sec, out_path, video_fps, padding_sec=0.25):
    """Re-read the video and write frames for [start_sec-pad, end_sec+pad]."""
    cap = cv2.VideoCapture(str(video_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or video_fps

    t_start = max(0.0, start_sec - padding_sec)
    t_end   = end_sec + padding_sec

    start_frame = int(t_start * fps)
    end_frame   = int(t_end   * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    fi = start_frame
    while fi <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def extract_matching_segments(video_path, query_images_root, output_dir,
                               threshold=0.80, fps=2, min_seconds=1.5,
                               gap_tolerance=2, padding_sec=0.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_model(device)

    # ── build query embedding matrix ──
    query_root = Path(query_images_root)
    folders = sorted([f for f in query_root.iterdir() if f.is_dir()])
    if not folders:
        print(f"No subfolders found in '{query_images_root}'.")
        return

    print(f"\nLoading query images from {len(folders)} folders...")
    query_emb_matrix, folder_map = embed_folder_images(folders, model, processor, device)
    if query_emb_matrix is None:
        print("No images found in query_images.")
        return

    # ── score every sampled frame ──
    print(f"\nScoring video frames @ {fps} FPS against {query_emb_matrix.shape[0]} query images...")
    frame_scores, video_fps = score_video_frames(
        video_path, query_emb_matrix, folder_map, model, processor, device, fps=fps
    )

    # ── find contiguous matching segments ──
    min_frames = max(1, int(min_seconds * fps))
    print(f"\nFinding segments: threshold={threshold}, min_duration={min_seconds}s, "
          f"gap_tolerance={gap_tolerance} samples")
    segments = find_segments(frame_scores, threshold, min_frames, gap_tolerance)

    # ── print results / plot ──
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem[:30]
    
    plot_path = Path(output_dir) / f"{video_stem}_timeline.png"
    save_timeline_plot(frame_scores, segments, threshold, plot_path)

    if not segments:
        print(f"\nNo segments found above threshold {threshold}.")
        scores = [f["score"] for f in frame_scores]
        print(f"Score distribution: mean={np.mean(scores):.4f}  max={np.max(scores):.4f}  "
              f"p95={np.percentile(scores, 95):.4f}")
        return

    print(f"\nFound {len(segments)} matching segment(s):\n")
    print(f"{'#':<4} {'Start':>8} {'End':>8} {'Duration':>9} {'Avg Score':>10} {'Best Match'}")
    print("─" * 70)
    for i, seg in enumerate(segments, 1):
        dur = seg["end_sec"] - seg["start_sec"]
        print(f"{i:<4} {seg['start_sec']:>7.2f}s {seg['end_sec']:>7.2f}s "
              f"{dur:>8.2f}s {seg['avg_score']:>10.4f}   {seg['best_folder']}")

    # ── crop and save segments ──
    print(f"\nSaving segments to '{output_dir}'...")
    for i, seg in enumerate(segments, 1):
        out_name = f"{video_stem}_seg{i:03d}_{seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s.mp4"
        out_path = Path(output_dir) / out_name
        crop_segment(video_path, seg["start_sec"], seg["end_sec"],
                     out_path, video_fps, padding_sec=padding_sec)
        print(f"  [{i}] {out_name} (avg={seg['avg_score']:.3f})")

    print(f"\nDone. {len(segments)} segment(s) saved to '{output_dir}'.")


    # ── also dump frame scores as CSV for further analysis ──
    csv_path = Path(output_dir) / f"{video_stem}_frame_scores.csv"
    with open(csv_path, "w") as f:
        f.write("frame_idx,time_sec,score\n")
        for fs in frame_scores:
            f.write(f"{fs['frame_idx']},{fs['time_sec']:.4f},{fs['score']:.6f}\n")
    print(f"Frame scores CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract matching video segments based on similarity to query images."
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--query_images", default="query_images",
        help="Root folder with per-video query image subfolders (default: query_images)"
    )
    parser.add_argument(
        "--output", default="segments",
        help="Output directory for cropped segments (default: segments)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Cosine similarity threshold for a frame to count as a match (default: 0.80)"
    )
    parser.add_argument(
        "--fps", type=float, default=2,
        help="Frames per second to sample from the video (default: 2)"
    )
    parser.add_argument(
        "--min_seconds", type=float, default=1.5,
        help="Minimum duration (seconds) for a segment to be kept (default: 1.5)"
    )
    parser.add_argument(
        "--gap_tolerance", type=int, default=2,
        help="How many consecutive below-threshold samples are allowed inside a segment before it is cut (default: 2)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.25,
        help="Extra seconds to include before/after each segment (default: 0.25)"
    )
    args = parser.parse_args()

    extract_matching_segments(
        video_path        = args.video,
        query_images_root = args.query_images,
        output_dir        = args.output,
        threshold         = args.threshold,
        fps               = args.fps,
        min_seconds       = args.min_seconds,
        gap_tolerance     = args.gap_tolerance,
        padding_sec       = args.padding,
    )
