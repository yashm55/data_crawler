"""
youtube_search_from_images.py
──────────────────────────────
Uses CLIP zero-shot matching to generate a YouTube search query from query
images, then uses yt-dlp to fetch the top 10 results (videos + shorts).

Usage:
    python youtube_search_from_images.py --folder query_images/my_video
    python youtube_search_from_images.py --folder query_images/my_video --top 20
    python youtube_search_from_images.py --query_root query_images          # uses ALL folders
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel
from PIL import Image
import yt_dlp

# ─────────────────────────────────────────────────────────────────────────────
# Candidate concepts for zero-shot query building
# These cover a broad range of common video content on YouTube.
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATE_CONCEPTS = [
    # Sports
    "cricket", "football", "basketball", "tennis", "badminton", "volleyball",
    "swimming", "athletics", "running race", "cycling race", "boxing",
    "wrestling", "kabaddi", "hockey", "rugby", "golf", "baseball",

    # Action / stunts
    "skateboarding", "parkour", "BMX", "dirt bike stunts", "motocross",

    # Dance / performance
    "classical dance", "hip hop dance", "street dance", "bharatanatyam",
    "wedding dance", "music performance", "concert",

    # Nature / outdoors
    "wildlife safari", "birds flying", "ocean waves", "waterfall", "sunset",
    "mountain trek", "forest", "snow mountains",

    # Food
    "street food", "cooking", "recipe", "restaurant", "biryani",

    # Vehicles / transport
    "car racing", "train journey", "airplane takeoff", "boat ride",

    # People / lifestyle
    "crowd festival", "market", "village life", "city timelapse",
    "gym workout", "yoga", "meditation",

    # News / events
    "political rally", "protest", "award ceremony", "graduation",

    # Children
    "kids playing", "cartoon", "school event",

    # Animals
    "dog", "cat", "elephant", "tiger", "horse racing",

    # Scenes
    "indoor", "outdoor", "stadium", "beach", "desert",

    # Quality descriptors useful for search
    "highlights", "compilation", "best moments", "tutorial", "documentary",
    "vlog", "review", "interview", "live stream",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device, model_type="clip"):
    if model_type == "siglip":
        model_id = "google/siglip-base-patch16-224"
        print(f"Loading SigLIP model ({model_id})...")
        model = SiglipModel.from_pretrained(model_id).to(device)
        processor = SiglipProcessor.from_pretrained(model_id)
    elif model_type == "openclip":
        model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        print(f"Loading OpenCLIP model ({model_id})...")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    else:
        model_id = "openai/clip-vit-base-patch32"
        print(f"Loading OpenAI CLIP model ({model_id})...")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def load_images_from_folder(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [Image.open(p).convert("RGB") for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]


def embed_images(images, model, processor, device, batch_size=32):
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            feats = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            feats = out[0] if isinstance(out, tuple) else out
        if hasattr(model, "visual_projection") and hasattr(model.config, "projection_dim"):
            if feats.shape[-1] != model.config.projection_dim:
                feats = model.visual_projection(feats)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.vstack(all_embs)


def embed_texts(texts, model, processor, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)

    if isinstance(out, torch.Tensor):
        feats = out
    elif hasattr(out, "text_embeds") and out.text_embeds is not None:
        feats = out.text_embeds
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feats = out.pooler_output
    else:
        feats = out[0] if isinstance(out, tuple) else out

    if hasattr(model, "text_projection") and hasattr(model.config, "projection_dim"):
        if feats.shape[-1] != model.config.projection_dim:
            feats = model.text_projection(feats)

    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()


def generate_query_from_images(images, model, processor, device, label="images", top_k=5):
    """CLIP zero-shot: pick top_k concepts that best describe a list of PIL images."""
    if not images:
        return None, []

    print(f"\nAnalyzing {len(images)} {label}...")
    image_emb = embed_images(images, model, processor, device)       # (N, D)
    mean_image_emb = image_emb.mean(axis=0, keepdims=True)           # (1, D)

    # Calculate zero-shot overlap directly without template engineering
    text_emb = embed_texts(CANDIDATE_CONCEPTS, model, processor, device)
    scores = (mean_image_emb @ text_emb.T).squeeze()
    
    concept_scores = []
    # Handle edge case where only 1 concept exists
    if len(CANDIDATE_CONCEPTS) == 1:
        scores = [scores]
        
    for i, concept in enumerate(CANDIDATE_CONCEPTS):
        concept_scores.append((float(scores[i]), concept))

    concept_scores.sort(reverse=True)
    top_concepts = [c for _, c in concept_scores[:top_k]]
    query = " ".join(top_concepts)

    print(f"  Top concepts : {', '.join(top_concepts)}")
    print(f"  Search query : \"{query}\"")
    return query, concept_scores


def search_youtube(query, max_results=50):
    """
    Search YouTube and return up to max_results unique video metadata dicts.
    Fetches (max_results * 2 + 50) candidates so there is a buffer after
    deduplication by video ID.
    """
    fetch_n = max(max_results * 2 + 50, 100)
    print(f"\nSearching YouTube for: \"{query}\" (fetching up to {fetch_n} candidates)...")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    search_url = f"ytsearch{fetch_n}:{query}"
    results = []
    seen_ids = set()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
        if info and "entries" in info:
            for entry in info["entries"]:
                if entry is None:
                    continue

                vid_id = entry.get("id", "")
                if not vid_id or vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)

                url = (
                    entry.get("webpage_url")
                    or entry.get("url")
                    or f"https://www.youtube.com/watch?v={vid_id}"
                )

                results.append({
                    "id":       vid_id,
                    "title":    entry.get("title", "N/A"),
                    "channel":  entry.get("uploader") or entry.get("channel", "N/A"),
                    "duration": entry.get("duration"),
                    "views":    entry.get("view_count"),
                    "url":      url,
                })

                if len(results) >= max_results:
                    break

    print(f"  Found {len(results)} unique results.")
    return results


def search_youtube_multi(queries, max_per_query=200, total_max=1000):
    """
    Search YouTube with multiple queries and return a deduplicated pool.
    Results are interleaved round-robin so no single query dominates.
    """
    per_query_results = []
    for q in queries:
        results = search_youtube(q, max_results=max_per_query)
        per_query_results.append(results)
        print(f"  Query \"{q[:50]}\": {len(results)} results")

    seen_ids = set()
    merged = []
    max_len = max((len(r) for r in per_query_results), default=0)

    for i in range(max_len):
        for bucket in per_query_results:
            if i < len(bucket):
                vid = bucket[i]
                vid_id = vid.get("id") or vid["url"]
                if vid_id not in seen_ids:
                    seen_ids.add(vid_id)
                    merged.append(vid)
                    if len(merged) >= total_max:
                        break
        if len(merged) >= total_max:
            break

    print(f"\n  Total unique candidates after multi-query merge: {len(merged)}")
    return merged


def print_results(results):
    print(f"\n{'─'*80}")
    print(f"{'#':<4} {'Title':<45} {'Channel':<20} {'Duration':>9} {'Views':>10}")
    print(f"{'─'*80}")
    for i, r in enumerate(results, 1):
        title    = (r["title"][:43] + "..") if len(r["title"]) > 45 else r["title"]
        channel  = (r["channel"][:18] + "..") if len(r["channel"]) > 20 else r["channel"]
        duration = f"{int(r['duration']//60)}:{int(r['duration']%60):02d}" if r["duration"] else "N/A"
        views    = f"{r['views']:,}" if r["views"] else "N/A"
        print(f"{i:<4} {title:<45} {channel:<20} {duration:>9} {views:>10}")
        print(f"     {r['url']}")
    print(f"{'─'*80}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a single YouTube search query from all query images using CLIP.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder",      help="Single query image folder (e.g. query_images/my_video)")
    group.add_argument("--query_root",  help="Pool ALL images from all subfolders into one unified query (e.g. query_images)")
    parser.add_argument("--top",        type=int, default=10,  help="Number of YouTube results (default: 10)")
    parser.add_argument("--concepts",   type=int, default=5,   help="Number of top concepts to build the query from (default: 5)")
    parser.add_argument("--model",      choices=["clip", "siglip"], default="clip", help="AI Model to use (default: clip)")
    parser.add_argument("--out_file", help="Path to a text file to save the found YouTube URLs (one per line)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, processor = load_model(device, model_type=args.model)

    all_urls = []
    if args.folder:
        # Single folder → one query
        images = load_images_from_folder(args.folder)
        label  = f"images from [{Path(args.folder).name}]"
    else:
        # ALL folders → pool every image into one global query
        folders = sorted([f for f in Path(args.query_root).iterdir() if f.is_dir()])
        print(f"\nPooling images from {len(folders)} folders in '{args.query_root}'...")
        images = []
        for folder in folders:
            folder_images = load_images_from_folder(folder)
            images.extend(folder_images)
            print(f"  [{folder.name}] — {len(folder_images)} images")
        label = f"images across {len(folders)} folders"

    print(f"\nTotal images loaded: {len(images)}")
    query, _ = generate_query_from_images(images, model, processor, device, label=label, top_k=args.concepts)
    if not query:
        print("No images found.")
        return

    print(f"\n{'═'*80}")
    results = search_youtube(query, max_results=args.top)
    if results:
        print_results(results)
        all_urls.extend([r["url"] for r in results])
    else:
        print("  No results found.")

    if args.out_file and all_urls:
        with open(args.out_file, "w") as f:
            for url in all_urls:
                f.write(url + "\n")
        print(f"\nSaved {len(all_urls)} URLs to: {args.out_file}")


if __name__ == "__main__":
    main()
