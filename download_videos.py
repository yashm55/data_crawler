"""
download_videos.py
────────────────────
Downloads YouTube videos (regular or Shorts) from a list of URLs or a text file.
Uses yt-dlp for efficient downloading and format selection.

Usage:
    python download_videos.py --urls "url1" "url2"
    python download_videos.py --file links.txt --output_dir downloaded_videos
"""

import argparse
import os
from pathlib import Path
import yt_dlp

def download_video(url, output_dir):
    """Download a single video using yt-dlp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a safe filename template that only includes ID and extension if title is messy
    # restrictfilenames ensures no special characters or spaces.
    outtmpl = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": True,
        "restrictfilenames": True, # Clean filenames for Windows/FileSystem reliability
    }

    print(f"\nStarting download: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            if info:
                filename = ydl.prepare_filename(info)
                # handle extension change if merged to mp4
                if not Path(filename).exists():
                     filename = filename.rsplit('.', 1)[0] + '.mp4'
                
                if Path(filename).exists():
                    print(f"  Successfully downloaded: {filename}")
                    return filename
                else:
                    print(f"  Failed to confirm download file existence for: {url}")
            else:
                print(f"  Failed to extract info or download: {url}")
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos for the Data Crawler pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", nargs="+", help="One or more YouTube URLs")
    group.add_argument("--file", help="Path to a text file containing YouTube URLs (one per line)")
    parser.add_argument("--output_dir", default="downloaded_videos", help="Directory to save downloaded videos (default: downloaded_videos)")
    
    args = parser.parse_args()

    urls = []
    if args.urls:
        urls = args.urls
    elif args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            print(f"Error: File '{args.file}' not found.")
            return

    if not urls:
        print("No URLs to download.")
        return

    print(f"Found {len(urls)} URL(s). Starting downloads into '{args.output_dir}'...")
    
    downloaded_files = []
    for url in urls:
        result = download_video(url, args.output_dir)
        if result:
            downloaded_files.append(result)

    print(f"\n{'═'*50}")
    print(f"Download Summary:")
    print(f"  Attempted:  {len(urls)}")
    print(f"  Successful: {len(downloaded_files)}")
    print(f"{'═'*50}")

if __name__ == "__main__":
    main()
