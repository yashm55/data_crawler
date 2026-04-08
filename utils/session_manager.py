import os
import json
import re
from pathlib import Path
from datetime import datetime

SESSIONS_DIR = Path("sessions")

def slugify(text):
    """Convert string to a safe filesystem name."""
    text = str(text).lower().strip()
    return re.sub(r'[^\w\-]', '_', text)

def create_session(name):
    """Create a new session directory and metadata."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = slugify(name)
    
    # Handle duplicates by adding a timestamp
    session_path = SESSIONS_DIR / session_id
    if session_path.exists():
        session_id = f"{session_id}_{datetime.now().strftime('%H%M%S')}"
        session_path = SESSIONS_DIR / session_id
        
    session_path.mkdir(parents=True, exist_ok=True)
    
    # Create required subdirectories
    for subdir in ["uploads", "query_images", "embeddings", "downloads", "results"]:
        (session_path / subdir).mkdir(exist_ok=True)
        
    meta = {
        "id": session_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "model_type": "clip",
        "threshold": 0.85,
        "last_query": "",
        "stats": {
            "uploaded_folders": 0,
            "downloaded_videos": 0,
            "clips_extracted": 0
        }
    }
    
    save_meta(session_id, meta)
    return meta

def list_sessions():
    """Return a list of all sessions ordered by creation date."""
    if not SESSIONS_DIR.exists():
        return []
        
    sessions = []
    for d in SESSIONS_DIR.iterdir():
        if d.is_dir():
            meta = load_meta(d.name)
            if meta:
                sessions.append(meta)
                
    # Sort by created_at descending
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions

def load_meta(session_id):
    """Load session metadata."""
    meta_path = SESSIONS_DIR / session_id / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading meta for {session_id}: {e}")
    return None

def save_meta(session_id, meta):
    """Save session metadata."""
    meta_path = SESSIONS_DIR / session_id / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

def delete_session(session_id):
    """Delete a session entirely."""
    session_path = SESSIONS_DIR / session_id
    if session_path.exists():
        import shutil
        shutil.rmtree(session_path, ignore_errors=True)
        return True
    return False

def get_session_path(session_id, subdir=""):
    """Get path to a specific subdirectory within a session."""
    path = SESSIONS_DIR / session_id
    if subdir:
        path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
    return path

def save_results(session_id, results):
    """Save the list of processed video results to the session's results folder."""
    results_path = SESSIONS_DIR / session_id / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

def load_results(session_id):
    """Load the list of processed video results. Returns [] if missing."""
    results_path = SESSIONS_DIR / session_id / "results.json"
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results for {session_id}: {e}")
    return []
