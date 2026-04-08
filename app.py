"""
app.py
──────
Flask backend for the Data Crawler UI.
Handles query analysis, search generation, and automated crawling.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # load .env before anything else

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from pathlib import Path
import threading
import queue
import torch
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from werkzeug.utils import secure_filename

from utils import core_logic
from utils import session_manager
from utils import llm_agent
from utils.sample_frames import sample_all_videos
from utils.youtube_search_from_images import generate_query_from_images, search_youtube, search_youtube_multi

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB Limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS_STATUS = {}

def get_status(session_id):
    if not session_id: return None
    if session_id not in SESSIONS_STATUS:
        SESSIONS_STATUS[session_id] = {
            "status": "idle",
            "progress": 0,
            "msg": "Ready",
            "last_query": "",
            "results": [],
            "model_type": "clip",
            "session_id": session_id
        }
    return SESSIONS_STATUS[session_id]

def emit_status(session_id):
    socketio.emit('pipeline_update', get_status(session_id))

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    return jsonify({"sessions": session_manager.list_sessions()})

@app.route('/api/sessions', methods=['POST'])
def create_session():
    data = request.json
    name = data.get("name")
    if not name:
        return jsonify({"error": "Name required"}), 400
    meta = session_manager.create_session(name)
    get_status(meta["id"]) # Initialize state
    return jsonify(meta)

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_manager.delete_session(session_id):
        if session_id in SESSIONS_STATUS:
            del SESSIONS_STATUS[session_id]
        return jsonify({"status": "success"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/sessions/<session_id>/activate', methods=['POST'])
def activate_session(session_id):
    meta = session_manager.load_meta(session_id)
    if not meta:
        return jsonify({"error": "Session not found"}), 404
        
    status = get_status(session_id)
    status["model_type"] = meta.get("model_type", "clip")
    status["results"] = session_manager.load_results(session_id)
    
    report = meta.get("report")
    return jsonify({"status": "success", "meta": meta, "results": status["results"], "report": report})

@app.route('/api/sessions/<session_id>/update_settings', methods=['POST'])
def update_settings(session_id):
    data = request.json
    meta = session_manager.load_meta(session_id)
    if not meta:
        return jsonify({"error": "Session not found"}), 404
    
    if "limit" in data: meta["limit"] = data["limit"]
    if "threshold" in data: meta["threshold"] = data["threshold"]
    if "sampling_size" in data: meta["sampling_size"] = data["sampling_size"]
    if "model_type" in data: meta["model_type"] = data["model_type"]
    
    session_manager.save_meta(session_id, meta)
    return jsonify({"status": "success"})

@app.route('/api/sessions/<session_id>/regenerate_embeddings', methods=['POST'])
def regenerate_embeddings(session_id):
    data = request.json
    model_type = data.get("model_type", "clip")
    meta = session_manager.load_meta(session_id)
    if not meta:
        return jsonify({"error": "Session not found"}), 404
        
    meta["model_type"] = model_type
    session_manager.save_meta(session_id, meta)
    
    status = get_status(session_id)
    status["model_type"] = model_type
    
    def regen_task():
        try:
            status["status"] = "analyzing"
            status["msg"] = "Regenerating embeddings..."
            emit_status(session_id)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = core_logic.load_model(device, model_type=model_type)
            
            q_img = session_manager.get_session_path(session_id, "query_images")
            if not any(q_img.iterdir()):
                raise Exception("No query images found for session.")
                
            report = core_logic.get_confusion_matrix_data(
                q_img, model, processor, device,
                preview_dir_path=session_manager.get_session_path(session_id, "query_previews"),
                preview_url_pref=f"/api/sessions/{session_id}/static/query_previews"
            )
            
            # Save new embeddings
            query_roots = [f for f in q_img.iterdir() if f.is_dir()]
            query_matrix, folder_map = core_logic.embed_folder_images(query_roots, model, processor, device)
            core_logic.save_embeddings(session_manager.get_session_path(session_id), query_matrix, folder_map)
            
            meta = session_manager.load_meta(session_id)
            if meta:
                meta["report"] = report
                session_manager.save_meta(session_id, meta)
            
            status["status"] = "idle"
            status["msg"] = "Embeddings regenerated"
            socketio.emit('pipeline_update', {"status": "idle", "msg": "Embeddings regenerated", "report": report, "session_id": session_id})
        except Exception as e:
            print(f"DEBUG: Regen failed {e}")
            status["status"] = "idle"
            status["msg"] = f"Error: {e}"
            emit_status(session_id)
            
    threading.Thread(target=regen_task).start()
    return jsonify({"status": "started"})
    
@app.route('/api/sessions/<session_id>/static/<path:filename>')
def serve_session_static(session_id, filename):
    return send_from_directory(session_manager.get_session_path(session_id), filename)

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
    session_id = request.form.get('session_id')
    
    if not session_id:
        return jsonify({"error": "No active session."}), 400
        
    status = get_status(session_id)
    status["model_type"] = model_type
    
    meta = session_manager.load_meta(session_id)
    if meta:
        meta["model_type"] = model_type
        # Additive mapping - count unique folders
        session_manager.save_meta(session_id, meta)

    # Note: We NO LONGER clear q_vid here to support additive appends!
    q_vid = session_manager.get_session_path(session_id, "uploads")
    q_img = session_manager.get_session_path(session_id, "query_images")
    q_vid.mkdir(parents=True, exist_ok=True)
    
    # Only clear query_images because we will resample everything in q_vid anyway
    robust_clear_dir(q_img)

    # Save files preserving folder structure
    try:
        newly_uploaded_folders = set()
        for file in files:
            filename = file.filename
            if not filename: continue
            
            if '/' in filename or '\\' in filename:
                parts = filename.replace('\\', '/').split('/')
                safe_folder = secure_filename(parts[0])
                if not safe_folder: safe_folder = "unknown_folder"
                
                subfolder = q_vid / safe_folder
                subfolder.mkdir(exist_ok=True)
                file.save(subfolder / secure_filename(parts[-1]))
                newly_uploaded_folders.add(safe_folder)
            else:
                file.save(q_vid / secure_filename(filename))
        
        # Update meta with new total count
        meta = session_manager.load_meta(session_id)
        if meta:
            video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
            all_videos = [p for p in q_vid.rglob('*') if p.is_file() and p.suffix.lower() in video_extensions]
            meta["stats"]["uploaded_folders"] = len(all_videos)
            session_manager.save_meta(session_id, meta)
            
    except Exception as e:
        print(f"DEBUG: Save failed: {e}")
        return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

    def background_sample_task():
        status["status"] = "sampling"
        status["msg"] = "Sampling frames from all uploaded videos..."
        status["progress"] = 0
        emit_status(session_id)

        def sampling_progress(p):
            status["progress"] = int(p * 100)
            status["msg"] = f"Sampling... {int(p * 100)}%"
            emit_status(session_id)

        try:
            sample_all_videos(q_vid, q_img, n_frames=sampling_size, progress_callback=sampling_progress)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = core_logic.load_model(device, model_type=model_type)

            status["status"] = "analyzing"
            status["msg"] = "Analyzing query similarities..."
            status["progress"] = 90
            emit_status(session_id)
            
            report = core_logic.get_confusion_matrix_data(
                q_img, model, processor, device,
                model_type=model_type,
                preview_dir_path=session_manager.get_session_path(session_id, "query_previews"),
                preview_url_pref=f"/api/sessions/{session_id}/static/query_previews"
            )
            
            query_roots = [f for f in q_img.iterdir() if f.is_dir()]
            query_matrix, folder_map = core_logic.embed_folder_images(query_roots, model, processor, device)
            core_logic.save_embeddings(session_manager.get_session_path(session_id), query_matrix, folder_map)

            meta = session_manager.load_meta(session_id)
            if meta:
                meta["report"] = report
                session_manager.save_meta(session_id, meta)

            status["status"] = "idle"
            status["msg"] = "Processing complete."
            # Targeted emit with the report payload for UI update
            socketio.emit('pipeline_update', {
                "status": "idle", "msg": "Upload & sampling complete", 
                "report": report, "session_id": session_id
            })
            
        except Exception as e:
            status["status"] = "idle"
            status["msg"] = f"Error: {e}"
            emit_status(session_id)

    threading.Thread(target=background_sample_task).start()
    return jsonify({"status": "started"})
    
@app.route('/api/resample_queries', methods=['POST'])
def resample_queries():
    """Trigger frame resampling directly from existing query_videos/ uploads."""
    data = request.json
    session_id = data.get("session_id")
    sampling_size = int(data.get("sampling_size", 5))
    model_type = data.get("model_type", "clip")
    
    if not session_id:
        return jsonify({"error": "Session ID required"}), 400
        
    status = get_status(session_id)
    if status["status"] != "idle":
        return jsonify({"error": "Session is busy."}), 400
        
    status["model_type"] = model_type
    meta = session_manager.load_meta(session_id)
    if meta:
        meta["model_type"] = model_type
        session_manager.save_meta(session_id, meta)
        
    def resample_task():
        try:
            status["status"] = "sampling"
            status["msg"] = "Resampling existing query pool..."
            status["progress"] = 0
            emit_status(session_id)
            
            q_vid = session_manager.get_session_path(session_id, "uploads")
            q_img = session_manager.get_session_path(session_id, "query_images")
            robust_clear_dir(q_img) # Delete old images for fresh sampling
            
            def sampling_progress(p):
                status["progress"] = int(p * 100)
                status["msg"] = f"Sampling... {int(p * 100)}%"
                emit_status(session_id)
                
            sample_all_videos(q_vid, q_img, n_frames=sampling_size, progress_callback=sampling_progress)
            
            # Reanalyze embeddings
            status["status"] = "analyzing"
            status["msg"] = "Analyzing query similarities..."
            status["progress"] = 90
            emit_status(session_id)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = core_logic.load_model(device, model_type=model_type)
            
            report = core_logic.get_confusion_matrix_data(
                q_img, model, processor, device,
                model_type=model_type,
                preview_dir_path=session_manager.get_session_path(session_id, "query_previews"),
                preview_url_pref=f"/api/sessions/{session_id}/static/query_previews"
            )
            
            query_roots = [f for f in q_img.iterdir() if f.is_dir()]
            query_matrix, folder_map = core_logic.embed_folder_images(query_roots, model, processor, device)
            core_logic.save_embeddings(session_manager.get_session_path(session_id), query_matrix, folder_map)
            
            meta = session_manager.load_meta(session_id)
            if meta:
                meta["report"] = report
                session_manager.save_meta(session_id, meta)
                
            status["status"] = "idle"
            status["msg"] = "Resampling Complete."
            socketio.emit('pipeline_update', {"status": "idle", "msg": "Resampling Complete", "report": report, "session_id": session_id})
        except Exception as e:
            status["status"] = "idle"
            status["msg"] = f"Error: {e}"
            emit_status(session_id)
            
    threading.Thread(target=resample_task).start()
    return jsonify({"status": "started"})

@app.route('/api/analyze_threshold', methods=['POST'])
def analyze_threshold():
    """Unused fallback route, logic mostly within get_confusion_matrix_data now."""
    return jsonify({"error": "Deprecated"}), 400

@app.route('/api/generate_search_prompt', methods=['POST'])
def generate_search_prompt():
    """Zero-shot CLIP matching for keywords."""
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "No active session."}), 400
        
    query_root = session_manager.get_session_path(session_id, "query_images")
    if not query_root.exists() or not any(query_root.iterdir()):
        return jsonify({"error": "No query images in session."}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    status = get_status(session_id)
    model_type = status.get("model_type", "clip")
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
    session_id = data.get("session_id")
    query = data.get("query")
    limit = data.get("limit", 3)
    threshold = data.get("threshold", 0.82)
    fps = data.get("fps", 2)
    
    if not session_id: return jsonify({"error": "Session ID required"}), 400
    
    status = get_status(session_id)
    if status["status"] != "idle":
        return jsonify({"error": "Crawl already in progress."}), 400
        
    threading.Thread(target=run_crawl_pipeline, args=(session_id, query, limit, threshold, fps, status["model_type"])).start()
    return jsonify({"status": "started"})

@app.route('/api/resume_crawl', methods=['POST'])
def resume_crawl():
    data = request.json
    session_id = data.get("session_id")
    threshold = data.get("threshold", 0.82)
    fps = data.get("fps", 2)
    
    if not session_id: return jsonify({"error": "Session ID required"}), 400
    
    status = get_status(session_id)
    if status["status"] != "idle":
        return jsonify({"error": "Crawl already in progress."}), 400
        
    threading.Thread(target=resume_crawl_pipeline, args=(session_id, threshold, fps, status["model_type"])).start()
    return jsonify({"status": "started"})

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Engine
# ─────────────────────────────────────────────────────────────────────────────

def process_video_loop(session_id, video_queue, threshold, fps, model_type, total_videos):
    """The Consumer Thread: Extracs video frames as files are downloaded."""
    status = get_status(session_id)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = core_logic.load_model(device, model_type=model_type)
        
        query_matrix, folder_map = core_logic.load_embeddings(session_manager.get_session_path(session_id))
        if query_matrix is None:
            raise Exception("Embeddings not found for this session. Please upload samples or regenerate.")
            
        segments_root = session_manager.get_session_path(session_id, "results")
        segments_root.mkdir(parents=True, exist_ok=True)
        all_results = session_manager.load_results(session_id)
        
        processed_count = 0
        
        while True:
            # Block until a video is ready
            vid_path = video_queue.get()
            if vid_path is None: # Sentinel value signals end of queue
                break
                
            processed_count += 1
            status["msg"] = f"Extracting [{processed_count}/{total_videos}]: {vid_path.name}"
            
            def update_progress(p):
                status["progress"] = int(p * 100)
                emit_status(session_id)
                
            scores, video_fps = core_logic.score_video_streaming(
                vid_path, query_matrix, folder_map, model, processor, device, 
                fps=fps, progress_callback=update_progress
            )
            
            min_f = max(1, int(1.5 * fps))
            segments = core_logic.find_segments(scores, threshold, min_f)
            
            vid_out_dir = segments_root / vid_path.stem
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = vid_out_dir / "timeline.png"
            core_logic.generate_timeline_plot(scores, segments, threshold, plot_path)
            
            extracted_clips = []
            for j, seg in enumerate(segments):
                clip_name = f"seg_{j:03d}_{seg['start_sec']:.1f}s.mp4"
                clip_path = vid_out_dir / clip_name
                core_logic.crop_video_segment(vid_path, seg["start_sec"], seg["end_sec"], clip_path)
                extracted_clips.append(f"/api/sessions/{session_id}/static/results/{vid_path.stem}/{clip_name}")
            
            res_obj = {
                "filename": vid_path.name,
                "title": vid_path.name, # Usually we'd map this, but relying strictly on filename is bulletproof
                "segments": len(segments),
                "plot": f"/api/sessions/{session_id}/static/results/{vid_path.stem}/timeline.png",
                "clips": extracted_clips,
                "status": "passed" if segments else "failed"
            }
            all_results.append(res_obj)
            session_manager.save_results(session_id, all_results)
            
            meta = session_manager.load_meta(session_id)
            if meta:
                meta["stats"]["clips_extracted"] += len(segments)
                session_manager.save_meta(session_id, meta)
                
            status["results"] = all_results
            emit_status(session_id)
            video_queue.task_done()
            
        status["status"] = "idle"
        status["msg"] = "Pipeline Complete!"
        status["progress"] = 100
        emit_status(session_id)
        
    except Exception as e:
        status["status"] = "idle"
        status["msg"] = f"Extraction Error: {str(e)}"
        emit_status(session_id)


def run_crawl_pipeline(session_id, query, limit, threshold, fps, model_type):
    """The Producer Thread: Fetches YT results and initiates parallel queue."""
    status = get_status(session_id)
    try:
        meta = session_manager.load_meta(session_id)
        if meta:
            meta["last_query"] = query
            meta["threshold"] = threshold
            meta["limit"] = limit
            session_manager.save_meta(session_id, meta)
            
        status["status"] = "searching"
        status["msg"] = "Searching YouTube..."
        emit_status(session_id)
        
        results = search_youtube(query, max_results=50)
        if not results:
            status["status"] = "idle"
            status["msg"] = "No search results found for this query."
            emit_status(session_id)
            return

        status["status"] = "downloading"
        status["msg"] = f"Initiating Download Thread..."
        emit_status(session_id)
        
        from utils.download_videos import download_video
        download_dir = session_manager.get_session_path(session_id, "downloads")
        archive_path = session_manager.get_session_path(session_id) / "archive.txt"
        
        # Parallel Producer-Consumer Pattern
        video_queue = queue.Queue()
        extract_thread = threading.Thread(
            target=process_video_loop, 
            args=(session_id, video_queue, threshold, fps, model_type, limit)
        )
        extract_thread.start()
        
        download_count = 0
        for i, r in enumerate(results):
            if download_count >= limit:
                break
                
            # Note: This is silent to the UI if extraction has started overriding the message
            print(f"[{session_id}] Downloading {r['title']}...")
            path = download_video(r["url"], download_dir, archive_path=archive_path)
            if path:
                p = Path(path)
                video_queue.put(p)
                download_count += 1
                
                meta = session_manager.load_meta(session_id)
                if meta:
                    meta["stats"]["downloaded_videos"] += 1
                    session_manager.save_meta(session_id, meta)
        
        # Add Sentinel correctly ensuring process stops
        video_queue.put(None)
        
    except Exception as e:
        status["status"] = "idle"
        status["msg"] = f"Pipeline Error: {str(e)}"
        emit_status(session_id)


def resume_crawl_pipeline(session_id, threshold, fps, model_type):
    status = get_status(session_id)
    try:
        meta = session_manager.load_meta(session_id)
        if meta:
            meta["threshold"] = threshold
            session_manager.save_meta(session_id, meta)
            
        downloads_dir = session_manager.get_session_path(session_id, "downloads")
        all_results = session_manager.load_results(session_id)
        
        # 1. Identify processed stems
        processed_stems = set()
        for r in all_results:
            if "filename" in r:
                processed_stems.add(Path(r["filename"]).stem)
            elif "plot" in r and "timeline.png" in r["plot"]:
                processed_stems.add(r["plot"].split("/")[-2])
                
        # 2. Find unprocessed downloads
        unprocessed_paths = []
        if downloads_dir.exists():
            for p in downloads_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi"}:
                    if p.stem not in processed_stems:
                        unprocessed_paths.append(p)
                        
        if not unprocessed_paths:
            status["status"] = "idle"
            status["msg"] = "No unprocessed videos found in downloads."
            emit_status(session_id)
            return

        status["status"] = "analyzing"
        status["msg"] = f"Found {len(unprocessed_paths)} videos. Resuming queue..."
        emit_status(session_id)
        
        video_queue = queue.Queue()
        for p in unprocessed_paths:
            video_queue.put(p)
        video_queue.put(None)
        
        extract_thread = threading.Thread(
            target=process_video_loop, 
            args=(session_id, video_queue, threshold, fps, model_type, len(unprocessed_paths))
        )
        extract_thread.start()

    except Exception as e:
        status["status"] = "idle"
        status["msg"] = f"Error: {str(e)}"
        emit_status(session_id)


@app.route('/api/start_agentic_crawl', methods=['POST'])
def start_agentic_crawl():
    """LLM-guided crawl: auto query generation, batch scoring, parallel downloads."""
    data = request.json or {}
    session_id = data.get("session_id")
    limit       = int(data.get("limit", 10))
    max_attempts = int(data.get("max_attempts", 3))

    if not session_id:
        return jsonify({"error": "Session ID required"}), 400

    status = get_status(session_id)
    if status["status"] != "idle":
        return jsonify({"error": "Session is busy."}), 400

    model_type = status.get("model_type", "clip")
    threading.Thread(
        target=run_agentic_pipeline,
        args=(session_id, limit, max_attempts, model_type),
        daemon=True,
    ).start()
    return jsonify({"status": "started"})


def _emit_s(session_id, msg, status_val=None, progress=None, llm_log=None):
    """Update session status and broadcast."""
    s = get_status(session_id)
    s["msg"] = msg
    if status_val is not None:
        s["status"] = status_val
    if progress is not None:
        s["progress"] = progress
    if llm_log is not None:
        s.setdefault("llm_log", []).append(llm_log)
    emit_status(session_id)


def run_agentic_pipeline(session_id, limit, max_attempts, model_type):
    """
    Session-aware agentic crawl.

    Phase 1 — Intelligence: BLIP captions + CLIP concepts → Claude describes
               target and generates N diverse search queries. Smart threshold
               computed from intra/inter-class similarity gap.

    Phase 2 — Wide search: all LLM queries searched at once via
               search_youtube_multi → large candidate pool. Claude batch-scores
               every candidate (20 at a time) and keeps the top `limit`.

    Phase 3 — Parallel download + sequential scoring: 4 concurrent download
               workers feed a queue; scoring runs as each download finishes.

    Phase 4 — Reflection: Claude evaluates yield and optionally refines
               the query and threshold for the next attempt.
    """
    status = get_status(session_id)
    status["llm_log"] = []

    try:
        from utils.download_videos import download_video

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = core_logic.load_model(device, model_type=model_type)

        q_img   = session_manager.get_session_path(session_id, "query_images")
        dl_dir  = session_manager.get_session_path(session_id, "downloads")
        seg_dir = session_manager.get_session_path(session_id, "results")
        seg_dir.mkdir(parents=True, exist_ok=True)
        archive = session_manager.get_session_path(session_id) / "archive.txt"

        query_folders = [f for f in q_img.iterdir() if f.is_dir()]

        # ── Phase 1: Intelligence ────────────────────────────────────────────
        _emit_s(session_id, "Sampling captions from query images...", "analyzing", 5)

        all_imgs = []
        for f in query_folders:
            all_imgs.extend(core_logic.load_folder_images(f))

        if not all_imgs:
            _emit_s(session_id, "No query images in session.", "idle", 0)
            return

        blm, blp = core_logic.load_blip_model(device)
        captions, step = [], max(1, len(all_imgs) // 5)
        for i in range(0, min(len(all_imgs), 5 * step), step):
            cap = core_logic.generate_blip_caption(all_imgs[i], blm, blp, device)
            if cap and cap not in captions:
                captions.append(cap)

        concepts = core_logic.generate_concepts_from_images(all_imgs, model, processor, device)

        # Smart threshold from per-folder embeddings
        folder_embs = {}
        for f in query_folders:
            imgs = core_logic.load_folder_images(f)
            if imgs:
                folder_embs[f.name] = core_logic._embed_batch(imgs, model, processor, device)

        current_threshold, thresh_diag = core_logic.compute_smart_threshold(folder_embs)
        _emit_s(session_id,
                f"Threshold: {current_threshold:.3f} (method: {thresh_diag.get('method')}, gap: {thresh_diag.get('gap')})",
                progress=10,
                llm_log=f"Threshold diagnostics: {thresh_diag}")

        _emit_s(session_id, "Claude: describing target content...", progress=12)
        query_description = llm_agent.describe_visual_content(captions, concepts)
        _emit_s(session_id, f"Target: {query_description}",
                llm_log=f"Description: {query_description}")

        n_queries = max_attempts + 1
        _emit_s(session_id, f"Claude: generating {n_queries} search queries...", progress=15)
        llm_queries = llm_agent.generate_search_queries(captions, concepts, n=n_queries)
        if not llm_queries:
            q, _ = generate_query_from_images(all_imgs, model, processor, device, top_k=5)
            llm_queries = [q] if q else ["video highlights"]
        _emit_s(session_id, f"Generated {len(llm_queries)} queries", progress=20,
                llm_log=f"Queries: {llm_queries}")

        # Load saved embeddings for scoring
        query_matrix, folder_map = core_logic.load_embeddings(session_manager.get_session_path(session_id))
        if query_matrix is None:
            _emit_s(session_id, "Embeddings not found. Please upload samples first.", "idle")
            return

        all_results = session_manager.load_results(session_id)
        remaining_queries = list(llm_queries)

        for attempt in range(1, max_attempts + 1):
            batch_queries = remaining_queries[:]
            remaining_queries = []

            _emit_s(session_id,
                    f"Attempt {attempt}/{max_attempts} — searching {len(batch_queries)} queries...",
                    "searching", 25,
                    llm_log=f"--- Attempt {attempt} | Queries: {batch_queries} ---")

            # ── Phase 2: Wide search + batch LLM scoring ─────────────────────
            candidates = search_youtube_multi(
                batch_queries,
                max_per_query=max(limit * 2, 100),
                total_max=limit * 3,
            )

            if not candidates:
                _emit_s(session_id, "No YouTube results found.", llm_log="0 results")
                break

            _emit_s(session_id,
                    f"Found {len(candidates)} candidates — Claude scoring...",
                    "filtering", 35)
            filtered, filter_reasoning = llm_agent.filter_videos_by_relevance(
                query_description, candidates, keep_n=limit
            )
            _emit_s(session_id,
                    f"Kept {len(filtered)}/{len(candidates)} after scoring",
                    progress=45,
                    llm_log=f"Filter: {filter_reasoning}")

            if not filtered:
                _emit_s(session_id, "No candidates passed scoring.", llm_log="0 kept")
                break

            # ── Phase 3: Parallel downloads + sequential scoring ─────────────
            _emit_s(session_id,
                    f"Downloading {len(filtered)} videos ({min(4, len(filtered))} workers)...",
                    "downloading", 48)

            completed_dl = 0
            downloaded = []   # list of (Path, title)

            def _dl(r):
                path = download_video(r["url"], dl_dir, archive_path=archive)
                return (Path(path), r["title"]) if path else None

            with ThreadPoolExecutor(max_workers=min(4, len(filtered))) as ex:
                futures = {ex.submit(_dl, r): r for r in filtered}
                for fut in as_completed(futures):
                    completed_dl += 1
                    result = fut.result()
                    _emit_s(session_id,
                            f"Downloaded {completed_dl}/{len(filtered)}...",
                            progress=48 + int(22 * completed_dl / len(filtered)))
                    if result:
                        downloaded.append(result)

            if not downloaded:
                _emit_s(session_id, "All downloads failed.", llm_log="0 downloaded")
                break

            attempt_results = []
            for i, (vid_path, title) in enumerate(downloaded):
                _emit_s(session_id,
                        f"Scoring {i+1}/{len(downloaded)}: {vid_path.name}",
                        "matching",
                        70 + int(15 * i / len(downloaded)))

                scores, _ = core_logic.score_video_streaming(
                    vid_path, query_matrix, folder_map, model, processor, device, fps=2
                )
                min_f = max(1, int(1.5 * 2))
                segments = core_logic.find_segments(scores, current_threshold, min_f)

                vid_out = seg_dir / vid_path.stem
                vid_out.mkdir(parents=True, exist_ok=True)
                core_logic.generate_timeline_plot(scores, segments, current_threshold, vid_out / "timeline.png")

                clips = []
                for j, seg in enumerate(segments):
                    clip_name = f"seg_{j:03d}_{seg['start_sec']:.1f}s.mp4"
                    core_logic.crop_video_segment(vid_path, seg["start_sec"], seg["end_sec"], vid_out / clip_name)
                    clips.append(f"/api/sessions/{session_id}/static/results/{vid_path.stem}/{clip_name}")

                res = {
                    "filename": vid_path.name,
                    "title":    title,
                    "segments": len(segments),
                    "plot":     f"/api/sessions/{session_id}/static/results/{vid_path.stem}/timeline.png",
                    "clips":    clips,
                    "status":   "passed" if segments else "failed",
                    "threshold_used": current_threshold,
                    "attempt":  attempt,
                }
                attempt_results.append(res)

                all_results.append(res)
                session_manager.save_results(session_id, all_results)

                meta = session_manager.load_meta(session_id)
                if meta:
                    meta["stats"]["clips_extracted"] += len(segments)
                    meta["stats"]["downloaded_videos"] += 1
                    session_manager.save_meta(session_id, meta)

                status["results"] = all_results
                emit_status(session_id)

            # ── Phase 4: Reflection ───────────────────────────────────────────
            _emit_s(session_id, "Claude: evaluating results...", progress=88)
            analysis = llm_agent.analyze_crawl_results(
                query_description, attempt_results,
                current_threshold, attempt, max_attempts,
            )
            assessment = analysis.get("assessment", "unknown")
            diagnosis  = analysis.get("diagnosis", "")
            _emit_s(session_id,
                    f"Claude [{assessment}]: {diagnosis}",
                    progress=92,
                    llm_log=(f"Assessment: {assessment} | {diagnosis}\n"
                             f"Retry: {analysis.get('should_retry')} | "
                             f"New query: {analysis.get('suggested_query')} | "
                             f"Δthreshold: {analysis.get('threshold_delta', 0.0)}"))

            if not analysis.get("should_retry") or attempt == max_attempts:
                stop = analysis.get("stop_reason") or (
                    "max attempts reached" if attempt == max_attempts else "yield is sufficient"
                )
                _emit_s(session_id, f"Stopping: {stop}", llm_log=f"Stop: {stop}")
                break

            if analysis.get("suggested_query"):
                remaining_queries.insert(0, analysis["suggested_query"])

            delta = float(analysis.get("threshold_delta", 0.0))
            if delta != 0.0:
                current_threshold = float(np.clip(current_threshold + delta, 0.70, 0.97))
                _emit_s(session_id,
                        f"Threshold → {current_threshold:.3f} (Δ={delta:+.3f})",
                        llm_log=f"Threshold adjusted to {current_threshold:.3f}")

        total_clips = sum(r["segments"] for r in all_results)
        status["status"]   = "idle"
        status["msg"]      = f"Done — {len(all_results)} videos, {total_clips} clips."
        status["progress"] = 100
        emit_status(session_id)

    except Exception as e:
        import traceback
        status["status"] = "idle"
        status["msg"]    = f"Agentic crawl error: {str(e)}"
        print(f"[agentic_pipeline] {traceback.format_exc()}")
        emit_status(session_id)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DATA CRAWLER DASHBOARD IS STARTING...")
    print("  URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')

