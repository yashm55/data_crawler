"""
app.py
──────
Flask backend for the Data Crawler UI.
Handles query analysis, search generation, and automated crawling.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from pathlib import Path
import threading
import torch
import numpy as np

import shutil
from werkzeug.utils import secure_filename
# Import our core logic
import core_logic
from sample_frames import sample_all_videos # We'll need this for query sampling
from youtube_search_from_images import generate_query_from_images, search_youtube

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB Limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

CRAWL_STATUS = {
    "status": "idle", # idle, sampling, analyzing, searching, crawling
    "progress": 0,
    "msg": "Ready",
    "last_query": "",
    "results": [],
    "model_type": "clip"
}

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/sample_queries', methods=['POST'])
def sample_queries():
    """Trigger frame sampling from query_videos/."""
    try:
        sample_all_videos()
        return jsonify({"status": "success", "msg": "Query images refreshed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def robust_clear_dir(p):
    """Reliably clear a directory even if some files are being held open (Windows-friendly)."""
    p = Path(p)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return
    
    # Try to rename instead of rmtree (often works even with open files)
    try:
        temp_p = p.parent / f".trash_{p.name}_{os.getpid()}"
        if temp_p.exists(): shutil.rmtree(temp_p, ignore_errors=True)
        p.rename(temp_p)
        shutil.rmtree(temp_p, ignore_errors=True)
    except:
        # Fallback to file-by-file deletion
        for item in p.iterdir():
            try:
                if item.is_dir(): shutil.rmtree(item, ignore_errors=True)
                else: item.unlink()
            except:
                pass # Skip if locked
    
    p.mkdir(parents=True, exist_ok=True)

@app.route('/api/upload_query_folders', methods=['POST'])
def upload_query_folders():
    """Handle multi-folder upload and auto-trigger preprocessing."""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
        
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files found"}), 400

    sampling_size = int(request.form.get('sampling_size', 5))
    model_type = request.form.get('model_type', 'clip')
    CRAWL_STATUS["model_type"] = model_type

    # Reset workspaces (using the robust helper)
    robust_clear_dir("uploads")
    robust_clear_dir("query_images")
    q_vid = Path("uploads")
    q_img = Path("query_images")

    # Save files preserving folder structure
    try:
        for file in files:
            filename = file.filename
            if not filename: continue
            
            if '/' in filename or '\\' in filename:
                parts = filename.replace('\\', '/').split('/')
                # Sanitize folder name too
                safe_folder = secure_filename(parts[0])
                if not safe_folder: safe_folder = "unknown_folder"
                
                subfolder = q_vid / safe_folder
                subfolder.mkdir(exist_ok=True)
                file.save(subfolder / secure_filename(parts[-1]))
            else:
                file.save(q_vid / secure_filename(filename))
    except Exception as e:
        print(f"DEBUG: Save failed: {e}")
        return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

    CRAWL_STATUS["status"] = "sampling"
    CRAWL_STATUS["msg"] = "Sampling frames from uploaded videos..."
    CRAWL_STATUS["progress"] = 0
    socketio.emit('pipeline_update', CRAWL_STATUS)

    def sampling_progress(p):
        CRAWL_STATUS["progress"] = int(p * 100)
        CRAWL_STATUS["msg"] = f"Sampling... {int(p * 100)}%"
        socketio.emit('pipeline_update', CRAWL_STATUS)

    try:
        sample_all_videos(q_vid, q_img, n_frames=sampling_size, progress_callback=sampling_progress)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = CRAWL_STATUS.get("model_type", "clip")
        model, processor = core_logic.load_model(device, model_type=model_type)

        # 3. Analyze Similarity
        CRAWL_STATUS["status"] = "analyzing"
        CRAWL_STATUS["msg"] = "Analyzing query similarities..."
        CRAWL_STATUS["progress"] = 90
        socketio.emit('pipeline_update', CRAWL_STATUS)
        report = core_logic.get_confusion_matrix_data(q_img, model, processor, device)

        CRAWL_STATUS["status"] = "idle"
        socketio.emit('pipeline_update', CRAWL_STATUS)

        return jsonify({
            "status": "success",
            "report": report
        })
    except Exception as e:
        CRAWL_STATUS["status"] = "idle"
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_threshold', methods=['POST'])
def analyze_threshold():
    """Run confusion matrix logic and suggest a threshold."""
    query_root = Path("query_images")
    if not query_root.exists():
        return jsonify({"error": "Query images folder not found."}), 400
    
    folders = [f for f in query_root.iterdir() if f.is_dir()]
    if not folders:
        return jsonify({"error": "No query folders found."}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = CRAWL_STATUS.get("model_type", "clip")
    model, processor = core_logic.load_model(device, model_type=model_type)
    
    # Simple threshold recommendation based on pairwise folder similarity
    # (Borrowing logic from query_confusion_matrix.py)
    names, mean_embs = [], []
    for f in folders:
        imgs = core_logic.load_folder_images(f)
        if imgs:
            emb = core_logic.embed_images(imgs, model, processor, device)
            mean_embs.append(emb.mean(axis=0, keepdims=True))
            names.append(f.name)
    
    if len(mean_embs) < 2:
        return jsonify({"recommended": 0.85, "msg": "Only one folder; default 0.85 recommended."})

    mat = np.vstack(mean_embs)
    sims = core_logic.cosine_similarity(mat, mat)
    # Get max off-diagonal
    off_diag = sims[~np.eye(len(names), dtype=bool)]
    max_sim = float(off_diag.max())
    recommended = round(max_sim + 0.05, 2)
    
    return jsonify({
        "recommended": recommended,
        "max_diff": max_sim,
        "msg": f"Max inter-folder similarity is {max_sim:.2f}; recommended {recommended}."
    })

@app.route('/api/generate_search_prompt', methods=['POST'])
def generate_search_prompt():
    """Zero-shot CLIP matching for keywords."""
    query_root = Path("query_images")
    if not query_root.exists():
        return jsonify({"error": "No query images."}), 400
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = CRAWL_STATUS.get("model_type", "clip")
    model, processor = core_logic.load_model(device, model_type=model_type)
    
    # Pool all images
    all_images = []
    for f in query_root.iterdir():
        if f.is_dir():
            all_images.extend(core_logic.load_folder_images(f))
    
    if not all_images:
        return jsonify({"error": "No images found."}), 400
        
    try:
        blm, blp = core_logic.load_blip_model(device)
        candidates = []
        step = max(1, len(all_images) // 5)
        for i in range(0, len(all_images), step):
            if len(candidates) >= 5: break
            cap = core_logic.generate_blip_caption(all_images[i], blm, blp, device)
            if cap and cap not in candidates:
                candidates.append(cap)
        
        return jsonify({"candidates": candidates})
    except Exception as e:
        print(f"BLIP manual failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_crawl', methods=['POST'])
def start_crawl():
    data = request.json
    query = data.get("query")
    limit = data.get("limit", 3)
    threshold = data.get("threshold", 0.82)
    fps = data.get("fps", 2)
    
    if CRAWL_STATUS["status"] != "idle":
        return jsonify({"error": "Crawl already in progress."}), 400
        
    model_type = CRAWL_STATUS.get("model_type", "clip")
    threading.Thread(target=run_crawl_pipeline, args=(query, limit, threshold, fps, model_type)).start()
    return jsonify({"status": "started"})

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_crawl_pipeline(query, limit, threshold, fps, model_type):
    global CRAWL_STATUS
    try:
        CRAWL_STATUS["status"] = "searching"
        CRAWL_STATUS["msg"] = "Searching YouTube..."
        socketio.emit('pipeline_update', CRAWL_STATUS)
        
        # 1. Search
        results = search_youtube(query, max_results=limit)
        if not results:
            CRAWL_STATUS["status"] = "idle"
            CRAWL_STATUS["msg"] = "No search results found for this query."
            socketio.emit('pipeline_update', CRAWL_STATUS)
            return

        print(f"DEBUG: Search found {len(results)} results (Requested limit: {limit})")
        CRAWL_STATUS["status"] = "downloading"
        CRAWL_STATUS["msg"] = f"Found {len(results)} videos. Starting downloads..."
        socketio.emit('pipeline_update', CRAWL_STATUS)
        
        # 2. Download
        from download_videos import download_video
        download_dir = Path("data_crawler_output/downloads")
        downloaded_paths = []
        video_titles = {} # Map filename to original title
        for i, r in enumerate(results):
            CRAWL_STATUS["msg"] = f"Downloading ({i+1}/{len(results)}): {r['title'][:30]}..."
            socketio.emit('pipeline_update', CRAWL_STATUS)
            
            path = download_video(r["url"], download_dir)
            if path:
                p = Path(path)
                downloaded_paths.append(p)
                video_titles[p.name] = r["title"]

        if not downloaded_paths:
             CRAWL_STATUS["status"] = "idle"
             CRAWL_STATUS["msg"] = "Downloads failed."
             socketio.emit('pipeline_update', CRAWL_STATUS)
             return

        # 3. Matching
        CRAWL_STATUS["status"] = "matching"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = core_logic.load_model(device, model_type=model_type)
        
        # Embed query images once
        query_roots = [f for f in Path("query_images").iterdir() if f.is_dir()]
        query_matrix, folder_map = core_logic.embed_folder_images(query_roots, model, processor, device)
        
        segments_root = Path("static/results")
        segments_root.mkdir(parents=True, exist_ok=True)
        
        final_results = []
        for i, vid_path in enumerate(downloaded_paths):
            CRAWL_STATUS["msg"] = f"Processing Video {i+1}/{len(downloaded_paths)}: {vid_path.name}"
            def update_progress(p):
                CRAWL_STATUS["progress"] = int(p * 100)
                socketio.emit('pipeline_update', CRAWL_STATUS)
            
            scores, video_fps = core_logic.score_video_streaming(
                vid_path, query_matrix, folder_map, model, processor, device, 
                fps=fps, progress_callback=update_progress
            )
            
            min_f = max(1, int(1.5 * fps))
            segments = core_logic.find_segments(scores, threshold, min_f)
            
            # Save segments and plot
            vid_out_dir = segments_root / vid_path.stem
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = vid_out_dir / "timeline.png"
            core_logic.generate_timeline_plot(scores, segments, threshold, plot_path)
            
            extracted_clips = []
            for j, seg in enumerate(segments):
                clip_name = f"seg_{j:03d}_{seg['start_sec']:.1f}s.mp4"
                clip_path = vid_out_dir / clip_name
                core_logic.crop_video_segment(vid_path, seg["start_sec"], seg["end_sec"], clip_path)
                extracted_clips.append(f"/static/results/{vid_path.stem}/{clip_name}")
            
            final_results.append({
                "title": video_titles.get(vid_path.name, vid_path.name),
                "segments": len(segments),
                "plot": f"/static/results/{vid_path.stem}/timeline.png",
                "clips": extracted_clips,
                "status": "passed" if segments else "failed"
            })
        
        CRAWL_STATUS["results"] = final_results
        CRAWL_STATUS["status"] = "idle"
        CRAWL_STATUS["msg"] = "Crawl Complete!"
        CRAWL_STATUS["progress"] = 100
        socketio.emit('pipeline_update', CRAWL_STATUS)

    except Exception as e:
        CRAWL_STATUS["status"] = "idle"
        CRAWL_STATUS["msg"] = f"Error: {str(e)}"
        socketio.emit('pipeline_update', CRAWL_STATUS)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DATA CRAWLER DASHBOARD IS STARTING...")
    print("  URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')

