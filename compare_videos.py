import cv2
import torch
import numpy as np
import argparse
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def extract_frames(video_path, fps="all"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
        
    if fps == "all":
        frame_interval = 1
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # Avoid division by zero
        if video_fps <= 0:
            video_fps = 30
        frame_interval = int(video_fps / fps)
        if frame_interval <= 0:
            frame_interval = 1
        
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            
        frame_count += 1
        
    cap.release()
    return frames

def get_embeddings(frames, model, processor, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Extracting embeddings"):
        batch_frames = frames[i:i + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            if isinstance(outputs, torch.Tensor):
                image_features = outputs
            elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                image_features = outputs.image_embeds
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                image_features = outputs.pooler_output
            else:
                image_features = outputs[0] if isinstance(outputs, tuple) else outputs
                
            # Some versions return unprojected pooler output wrapped in an object; we project if needed
            if hasattr(model, 'visual_projection') and hasattr(model.config, 'projection_dim'):
                if image_features.shape[-1] != model.config.projection_dim:
                    image_features = model.visual_projection(image_features)
                    
            # Normalize the features for cosine similarity calculation
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
            
    return np.vstack(embeddings) if embeddings else np.empty((0, model.config.projection_dim))

def compare_videos(video1_path, video2_path, output_plot="similarity_matrix.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model and processor
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    print(f"\nExtracting all frames from {video1_path}...")
    frames1 = extract_frames(video1_path, fps="all")
    
    print(f"Extracting all frames from {video2_path}...")
    frames2 = extract_frames(video2_path, fps="all")
    
    if not frames1 or not frames2:
        print("Error: Could not extract frames from one or both videos.")
        return
        
    print(f"Extracted {len(frames1)} frames from Video 1 and {len(frames2)} frames from Video 2.")
    
    # Get embeddings
    print("\nCalculating embeddings for Video 1...")
    emb1 = get_embeddings(frames1, model, processor, device)
    
    print("\nCalculating embeddings for Video 2...")
    emb2 = get_embeddings(frames2, model, processor, device)
    
    # Calculate cosine similarity matrix
    print("\nCalculating cosine similarity matrix...")
    sim_matrix = cosine_similarity(emb1, emb2)
    
    # Find the best match for each frame in video 1
    best_matches = np.argmax(sim_matrix, axis=1)
    best_scores = np.max(sim_matrix, axis=1)
    
    avg_best_score = np.mean(best_scores)
    max_score = np.max(best_scores)
    
    print(f"\n--- Results ---")
    print(f"Average best similarity score: {avg_best_score:.4f}")
    print(f"Maximum similarity score: {max_score:.4f}")
    
    # Plot the similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Frame-wise Cosine Similarity between Two Videos')
    plt.xlabel('Video 2 Frames (seconds)')
    plt.ylabel('Video 1 Frames (seconds)')
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved similarity matrix plot to {output_plot}")
    
    # Optionally print some top matches
    print("\nTop 5 frame matches (Video 1 Frame -> Video 2 Frame = Score):")
    top_indices = np.argsort(best_scores)[::-1][:5]
    for idx in top_indices:
        print(f"V1 {idx}s -> V2 {best_matches[idx]}s = {best_scores[idx]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two videos using CLIP embeddings based cosine similarity.")
    parser.add_argument("video1", help="Path to the first video")
    parser.add_argument("video2", help="Path to the second video")
    parser.add_argument("--output", default="similarity_matrix.png", help="Path to save the similarity matrix plot")
    
    args = parser.parse_args()
    compare_videos(args.video1, args.video2, args.output)
