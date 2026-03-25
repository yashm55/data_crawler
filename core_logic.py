"""
core_logic.py
──────────────
Contains the shared CLIP and segment matching logic for the Data Crawler pipeline.
Used by app.py and CLI tools.
"""

import cv2
import torch
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel, BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Model & Embedding
# ─────────────────────────────────────────────────────────────────────────────

# Cache the loaded model to avoid re-loading
_LOADED_MODEL = {"type": None, "model": None, "processor": None}

def load_model(device, model_type="clip"):
    global _LOADED_MODEL
    if _LOADED_MODEL["type"] == model_type and _LOADED_MODEL["model"] is not None:
        return _LOADED_MODEL["model"], _LOADED_MODEL["processor"]

    print(f"Loading model: {model_type} on {device}...")
    if model_type == "siglip":
        model_id = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_id).to(device)
        processor = SiglipProcessor.from_pretrained(model_id)
    elif model_type == "openclip":
        model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        model = AutoModel.from_pretrained(model_id).to(device)
        processor = AutoImageProcessor.from_pretrained(model_id)
    elif model_type == "dinov2":
        model_id = "facebook/dinov2-base"
        model = AutoModel.from_pretrained(model_id).to(device)
        processor = AutoImageProcessor.from_pretrained(model_id)
    else:
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    
    _LOADED_MODEL = {"type": model_type, "model": model, "processor": processor}
    return model, processor
_BLIP_MODEL = {"model": None, "processor": None}

def load_blip_model(device):
    global _BLIP_MODEL
    if _BLIP_MODEL["model"] is not None:
        return _BLIP_MODEL["model"], _BLIP_MODEL["processor"]
    
    print(f"Loading BLIP captioning model on {device}...")
    model_id = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    processor = BlipProcessor.from_pretrained(model_id)
    _BLIP_MODEL = {"model": model, "processor": processor}
    return model, processor

def generate_blip_caption(pil_image, model, processor, device):
    """Generate a natural language caption for a single image."""
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_concepts_from_images(pil_images, model, processor, device):
    """Manual zero-shot matching bypassing fragile Transformers wrappers."""
    # Ensure it's a model that supports text matching
    if not hasattr(model, "get_text_features"):
        return ["Visual Features Only"]

    labels = [
        "sports", "news", "educational", "vlog", "gaming", "coding", "cricket", "football",
        "action", "slow motion", "interview", "tutorial", "cinematic", "documentary", 
        "highlights", "commentary", "outdoor", "indoor", "urban", "nature", "person", "crowd"
    ]
    
    try:
        # 1. Get Text Features
        tok = getattr(processor, "tokenizer", processor)
        txt_inputs = tok(text=labels, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_feats = model.get_text_features(**txt_inputs)
            text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        
        # 2. Get Image Features (Reuse our robust embedder)
        img_feats_list = []
        for i in range(min(len(pil_images), 5)):
             img_feats_list.append(_embed_batch([pil_images[i]], model, processor, device))
        
        # Average image features
        img_feats = np.mean(np.vstack(img_feats_list), axis=0, keepdims=True)
        img_feats_torch = torch.from_numpy(img_feats).to(device)
        
        # 3. Match
        # (1, D) @ (N_labels, D).T -> (1, N_labels)
        logits = (img_feats_torch @ text_feats.T).cpu().numpy()[0]
        idxs = np.argsort(logits)[-5:][::-1]
        return [labels[i] for i in idxs]
        
    except Exception as e:
        print(f"DEBUG: Manual concept matching failed: {e}")
        return ["AI Insight Unavailable"]

def load_folder_images(folder_path):
    """Load all images from a folder as PIL objects."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder = Path(folder_path)
    return [Image.open(p).convert("RGB") for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]

def _embed_batch(pil_images, model, processor, device):
    """Embed a list of PIL images and return normalized numpy array."""
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            out = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
        
        # Robustly extract Tensor features from output object
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
             feats = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
             feats = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
             feats = out.last_hidden_state[:, 0, :]
        elif isinstance(out, (list, tuple)):
             feats = out[0]
        else:
            # Final fallback
            feats = getattr(out, "last_hidden_state", getattr(out, "pooler_output", out))
            if hasattr(feats, "__getitem__") and not isinstance(feats, torch.Tensor):
                feats = feats[0]
        
        # Critical: Final check to ensure we have a Tensor for later processing
        if not isinstance(feats, torch.Tensor):
            print(f"DEBUG: Unresolved type {type(out)}. Attempting forced tensor conversion.")
            # Forcing to first element or dict value
            if isinstance(out, dict): feats = list(out.values())[0]
            else: feats = out[0]

    if hasattr(model, "visual_projection") and hasattr(model.config, "projection_dim"):
        if feats.shape[-1] != model.config.projection_dim:
            feats = model.visual_projection(feats)

    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()

def embed_folder_images(folder_paths, model, processor, device, batch_size=32):
    """Embed all images from multiple folders, return matrix and folder mapping."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []
    folder_map = []
    
    for folder in folder_paths:
        folder = Path(folder)
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in exts:
                try:
                    all_images.append(Image.open(p).convert("RGB"))
                    folder_map.append(folder.name)
                except:
                    pass

    if not all_images:
        return None, None
    
    all_embs = []
    for i in range(0, len(all_images), batch_size):
        all_embs.append(_embed_batch(all_images[i:i+batch_size], model, processor, device))
    
    emb_matrix = np.vstack(all_embs)
    return emb_matrix, folder_map

def get_confusion_matrix_data(query_root_path, model, processor, device):
    """Calculate pairwise folder similarity and return structured data for UI."""
    folders = sorted([f for f in Path(query_root_path).iterdir() if f.is_dir()])
    if not folders:
        return None
    
    names, mean_embs = [], []
    for f in folders:
        imgs = load_folder_images(f)
        if imgs:
            emb = _embed_batch(imgs, model, processor, device)
            mean_embs.append(emb.mean(axis=0, keepdims=True))
            names.append(f.name)
            
    if not names:
        return None
        
    mat = np.vstack(mean_embs)
    sims = cosine_similarity(mat, mat)
    
    off_diag = sims[~np.eye(len(names), dtype=bool)]
    if off_diag.size == 0:
        return {
            "names": names,
            "max_sim": 1.0,
            "min_sim": 1.0,
            "mean_sim": 1.0,
            "recommended_threshold": 0.85
        }
        
    max_sim = float(off_diag.max())
    min_sim = float(off_diag.min())
    mean_sim = float(off_diag.mean())
    recommended = min(0.98, round(mean_sim, 2))
    
    # NEW: Identify 'distinct' folders based on similarity
    distinct_names = []
    distinct_indices = []
    for i, name in enumerate(names):
        is_distinct = True
        for prev_idx in distinct_indices:
            if sims[i, prev_idx] > recommended:
                is_distinct = False
                break
        if is_distinct:
            distinct_names.append(name)
            distinct_indices.append(i)
    
    # Generate Global Detected Trends from ALL images
    all_all_imgs = []
    for name in distinct_names:
        all_all_imgs.extend(load_folder_images(Path("query_images") / name))
    
    trends = generate_concepts_from_images(all_all_imgs, model, processor, device) if all_all_imgs else []

    for name in distinct_names:
        folder_path = Path("query_images") / name
        imgs = sorted(list(folder_path.glob("*.jpg")))
        if imgs:
            dest = preview_dir / f"{name}.jpg"
            shutil.copy(imgs[0], dest)
            thumbnails.append(f"/static/query_previews/{name}.jpg")

    return {
        "names": names,
        "matrix": sims.tolist(),
        "max_sim": max_sim,
        "min_sim": min_sim,
        "mean_sim": mean_sim,
        "recommended_threshold": recommended,
        "thumbnails": thumbnails,
        "concepts": [], # Removed per user 
        "trends": trends, # Global summary
        "distinct_names": distinct_names
    }

# ─────────────────────────────────────────────────────────────────────────────
# Segmentation & Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_video_streaming(video_path, query_emb_matrix, folder_map, model, processor, device,
                        fps=2, batch_size=16, progress_callback=None):
    """
    Stream video, compute per-frame similarity against query images.
    Returns: frame_scores (list of dicts), video_fps
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 30.0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(video_fps / fps))) if fps > 0 else 1
    
    frame_scores = []
    batch_pil, batch_meta = [], []
    frame_idx = 0
    sampled_count = 0
    n_expected = total_frames // interval

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
                sims = cosine_similarity(embs, query_emb_matrix)
                scores = sims.max(axis=1)
                best_indices = sims.argmax(axis=1)
                
                for meta, score, b_idx in zip(batch_meta, scores, best_indices):
                    frame_scores.append({**meta, "score": float(score), "best_folder": folder_map[b_idx]})
                
                batch_pil.clear(); batch_meta.clear()
                sampled_count += batch_size
                if progress_callback:
                    progress_callback(min(1.0, sampled_count / n_expected))
        
        frame_idx += 1
        del frame

    if batch_pil:
        embs = _embed_batch(batch_pil, model, processor, device)
        sims = cosine_similarity(embs, query_emb_matrix)
        scores = sims.max(axis=1)
        best_indices = sims.argmax(axis=1)
        for meta, score, b_idx in zip(batch_meta, scores, best_indices):
            frame_scores.append({**meta, "score": float(score), "best_folder": folder_map[b_idx]})
        if progress_callback:
            progress_callback(1.0)

    cap.release()
    return frame_scores, video_fps

def find_segments(frame_scores, threshold, min_frames, gap_tolerance=2):
    """Identify contiguous segments with score >= threshold."""
    segments = []
    current = None
    below_run = 0

    for fs in frame_scores:
        above = fs["score"] >= threshold
        if above:
            below_run = 0
            if current is None:
                current = {"start_frame": fs["frame_idx"], "start_sec": fs["time_sec"], "scores": []}
            current["scores"].append(fs["score"])
            current["end_frame"] = fs["frame_idx"]
            current["end_sec"] = fs["time_sec"]
            current["best_folder"] = fs["best_folder"]
        else:
            if current is not None:
                below_run += 1
                if below_run > gap_tolerance:
                    if len(current["scores"]) >= min_frames:
                        segments.append({
                            **current,
                            "avg_score": float(np.mean(current["scores"])),
                            "n_frames": len(current["scores"])
                        })
                    current = None
                    below_run = 0
    
    if current is not None and len(current["scores"]) >= min_frames:
        segments.append({**current, "avg_score": float(np.mean(current["scores"])), "n_frames": len(current["scores"])})
    
    return segments

# ─────────────────────────────────────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────────────────────────────────────

def crop_video_segment(video_path, start_sec, end_sec, out_path, padding_sec=0.25):
    """Crop a segment from a video file using FFmpeg for browser compatibility (H.264)."""
    t_start = max(0.0, float(start_sec) - padding_sec)
    duration = float(end_sec) + padding_sec - t_start
    
    # Use libx264 for H.264 encoding, yuv420p pixel format, and faststart for web playback
    cmd = [
        "ffmpeg", "-y", 
        "-ss", f"{t_start:.3f}", 
        "-t", f"{duration:.3f}", 
        "-i", str(video_path),
        "-vcodec", "libx264", 
        "-pix_fmt", "yuv420p", 
        "-crf", "23",           # High quality, small file size
        "-acodec", "aac", 
        "-b:a", "128k", 
        "-movflags", "+faststart", # Move metadata to the front (critical for browser playback)
        str(out_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cropping segment with FFmpeg: {e.stderr}")
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Visuals
# ─────────────────────────────────────────────────────────────────────────────

def generate_timeline_plot(frame_scores, segments, threshold, out_path):
    """Save a similarity vs time plot."""
    times = [fs["time_sec"] for fs in frame_scores]
    scores = [fs["score"] for fs in frame_scores]
    
    plt.figure(figsize=(12, 5))
    plt.plot(times, scores, label="Confidence", color="#2E86C1", alpha=0.7)
    plt.axhline(y=threshold, color="#E74C3C", linestyle="--", label="Threshold")
    
    for seg in segments:
        plt.axvspan(seg["start_sec"], seg["end_sec"], color="#27AE60", alpha=0.2)
        plt.text((seg["start_sec"] + seg["end_sec"])/2, max(scores)*1.02, seg["best_folder"], ha="center", fontsize=8)

    plt.xlabel("Time (s)")
    plt.ylabel("Similarity")
    plt.title("CLIP Match Confidence Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
