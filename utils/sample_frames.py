import cv2
import os
import numpy as np
from pathlib import Path

def sample_frames(video_path, output_dir, n_frames=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: Could not determine frame count for {video_path}")
        cap.release()
        return

    # Evenly spaced frame indices across the video
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    video_name = video_path.stem
    video_out_dir = output_dir / video_name
    video_out_dir.mkdir(exist_ok=True)

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = video_out_dir / f"frame{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print(f"  [{video_name}] saved {saved}/{n_frames} frames")

def sample_all_videos(query_videos_dir="query_videos", query_images_dir="query_images", n_frames=5, progress_callback=None):
    query_videos_dir = Path(query_videos_dir)
    query_images_dir = Path(query_images_dir)

    if not query_videos_dir.exists():
        print(f"Error: '{query_videos_dir}' folder not found.")
        return

    query_images_dir.mkdir(exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    # Find all videos recursively
    videos = []
    for root, _, files in os.walk(query_videos_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in video_extensions:
                videos.append(p)
    
    videos = sorted(videos)
    if not videos:
        print(f"No video files found in '{query_videos_dir}'.")
        return

    print(f"Found {len(videos)} video(s) in '{query_videos_dir}'. Sampling {n_frames} frames each...")
    for i, video in enumerate(videos):
        sample_frames(video, query_images_dir, n_frames=n_frames)
        if progress_callback:
            progress_callback((i + 1) / len(videos))

    print(f"\nDone! Frames saved to '{query_images_dir}'.")

def main():
    sample_all_videos()

if __name__ == "__main__":
    main()

