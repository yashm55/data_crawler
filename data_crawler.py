"""
data_crawler.py
────────────────
The main orchestrator for the Data Crawler pipeline.
1. Generates a search query from ALL images in query_images/
2. Searches YouTube for the top N results
3. Downloads the videos
4. Extracts matching segments from each downloaded video
5. Organizes the output

Usage:
    python data_crawler.py --query_root query_images --num_results 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_script(script_name, args):
    """Run a python script in the current virtual environment."""
    python_exe = str(Path("venv") / "Scripts" / "python.exe")
    if not Path(python_exe).exists():
        python_exe = sys.executable # Fallback to current interpreter
    
    cmd = [python_exe, script_name] + args
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="End-to-End Data Crawler Pipeline")
    parser.add_argument("--query_root", default="query_images", help="Folder containing query images (default: query_images)")
    parser.add_argument("--num_results", type=int, default=3, help="Number of YouTube search results to process (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.82, help="Similarity threshold for segment matching (default: 0.82)")
    parser.add_argument("--fps", type=float, default=2, help="FPS to sample for matching (default: 2)")
    parser.add_argument("--output_dir", default="data_crawler_output", help="Final output root directory")
    
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    download_dir = output_root / "downloads"
    segments_dir = output_root / "segments"
    search_results_file = output_root / "search_results.txt"
    
    print("\n" + "═"*60)
    print(" DATA CRAWLER PIPELINE STARTing")
    print("═"*60)

    # 1. Search YouTube
    print("\n[STEP 1/3] Searching YouTube using CLIP-generated query...")
    if not run_script("youtube_search_from_images.py", [
        "--query_root", args.query_root,
        "--top", str(args.num_results),
        "--out_file", str(search_results_file)
    ]):
        print("Search failed.")
        return

    if not search_results_file.exists() or search_results_file.stat().st_size == 0:
        print("No search results found to download.")
        return

    # 2. Download Videos
    print("\n[STEP 2/3] Downloading found videos...")
    if not run_script("download_videos.py", [
        "--file", str(search_results_file),
        "--output_dir", str(download_dir)
    ]):
        print("Download step reported issues.")
        # Proceed anyway as some might have succeeded

    # 3. Extract Segments
    print("\n[STEP 3/3] Extracting matching segments from downloaded videos...")
    downloaded_videos = list(download_dir.glob("*.mp4")) + list(download_dir.glob("*.mkv"))
    if not downloaded_videos:
        print("No videos found in download directory to process.")
        return

    print(f"Found {len(downloaded_videos)} video(s) to process.")
    for i, video_path in enumerate(downloaded_videos, 1):
        print(f"\n({i}/{len(downloaded_videos)}) Processing: {video_path.name}")
        run_script("extract_matching_segments.py", [
            str(video_path),
            "--query_images", args.query_root,
            "--output", str(segments_dir / video_path.stem),
            "--threshold", str(args.threshold),
            "--fps", str(args.fps)
        ])

    print("\n" + "═"*60)
    print(" DATA CRAWLER PIPELINE COMPLETE")
    print("═"*60)
    print(f"Final segments are in: {segments_dir}")


if __name__ == "__main__":
    main()
